"""Epoch/cursor resume checks; real Torch loader checks run when installed.

Toy tests establish the ordering/counter protocol only, never GPU, simulator,
optimizer or independent RLinf worker-resume evidence.
"""
import ast
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random
import tempfile
import types
import unittest

from xvla_rlinf.sft import EpochCursorSampler


class OrderedDistributedSampler:
    """Small deterministic ordering oracle, unrelated to model computation."""
    def __init__(self, size=17, *, rank=0):
        self.dataset=list(range(size))
        self.num_replicas=1
        self.rank=rank
        self.shuffle=True
        self.seed=1234
        self.drop_last=False
        self.epoch=0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch=epoch

    def __iter__(self):
        result=list(self.dataset)
        random.Random(self.seed+self.epoch).shuffle(result)
        return iter(result)


def toy_sampler(epoch=0, **kwargs):
    sampler=EpochCursorSampler(OrderedDistributedSampler(**kwargs))
    sampler.set_epoch(epoch)
    return sampler


class SamplerResumeTests(unittest.TestCase):
    def test_later_epoch_restores_cursor_and_full_epoch_length(self):
        sampler=toy_sampler(3)
        iterator=iter(sampler)
        first=list(itertools.islice(iterator,5))
        state=copy.deepcopy(sampler.state_dict())
        expected=list(iterator)
        restored=toy_sampler()
        restored.load_state_dict(state)
        self.assertEqual(restored.epoch,3)
        self.assertEqual(len(restored),17)
        self.assertEqual(list(restored),expected)
        self.assertEqual(len(set(first+expected)),17)

    def test_same_epoch_does_not_erase_pending_cursor(self):
        sampler=toy_sampler(2)
        iterator=iter(sampler)
        next(iterator)
        saved=sampler.state_dict()
        expected=list(iterator)
        restored=toy_sampler()
        restored.load_state_dict(saved)
        restored.set_epoch(2)
        self.assertEqual(list(restored),expected)
        # A completed, un-restored iterator restarts its complete epoch.
        self.assertEqual(len(list(restored)),17)

    def test_next_epoch_clears_old_cursor_and_changes_order(self):
        sampler=toy_sampler(2)
        iterator=iter(sampler)
        next(iterator)
        sampler.load_state_dict(sampler.state_dict())
        sampler.set_epoch(3)
        self.assertEqual(list(sampler),list(toy_sampler(3)))

    def test_finished_epoch_resume_remains_exhausted_until_next_epoch(self):
        sampler=toy_sampler(4)
        list(sampler)
        restored=toy_sampler()
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(list(restored),[])
        restored.set_epoch(5)
        self.assertEqual(list(restored),list(toy_sampler(5)))

    def test_identity_and_invalid_cursors_fail_closed(self):
        state=toy_sampler(1).state_dict()
        for restored in (toy_sampler(size=18),toy_sampler(rank=1)):
            with self.assertRaises(ValueError):
                restored.load_state_dict(state)
        for field,value in [('epoch',-1),('yielded',18),('yielded',True),('version',2)]:
            invalid=copy.deepcopy(state)
            invalid[field]=value
            with self.assertRaises(ValueError):
                toy_sampler().load_state_dict(invalid)


def worker_class_for_test(base):
    """Execute the actual checkpoint wrappers around a lightweight boundary."""
    source=Path(__file__).resolve().parents[2]/'xvla_rlinf/sft_entry.py'
    tree=ast.parse(source.read_text())
    worker=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='XVLASftWorker')
    worker.body=[node for node in worker.body if isinstance(node,ast.FunctionDef)
                 and node.name in ('_save_resume_counters','load_checkpoint')]
    namespace={'FSDPVlaSftWorker':base,'Path':Path,'json':json,
               'torch':types.SimpleNamespace(distributed=types.SimpleNamespace(
                   all_gather_object=lambda output,value:output.__setitem__(0,value),barrier=lambda:None))}
    exec(compile(ast.Module(body=[worker],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['XVLASftWorker']


class ToyLoader:
    def __init__(self,sampler):
        self.sampler=sampler
        self.sampler_epoch_at_load=None

    def __iter__(self):
        return iter(self.sampler)

    def load_state_dict(self,state):
        self.sampler_epoch_at_load=self.sampler.epoch
        self.sampler.load_state_dict(state)


class WorkerCounterResumeTests(unittest.TestCase):
    def test_real_wrapper_prepares_epoch_before_parent_iterator_and_restores_counters(self):
        original=toy_sampler(3)
        iterator=iter(original)
        list(itertools.islice(iterator,5))
        state=original.state_dict()
        expected=list(iterator)

        class NativeBoundary:
            def load_checkpoint(self,path):
                # This is the pinned parent's relevant call order.
                self.data_loader.load_state_dict(state)
                self.data_iter=iter(self.data_loader)

        Worker=worker_class_for_test(NativeBoundary)
        with tempfile.TemporaryDirectory() as directory:
            worker=Worker()
            worker._rank=0
            worker._world_size=1
            worker.data_config={'manifest_sha256':'fixed-demo-hash'}
            worker.micro_batch_size=1
            worker._data_epoch=3
            worker._data_iter_offset=5
            worker.optimizer_steps=56
            worker.data_loader=ToyLoader(toy_sampler(3))
            worker._audit_write=lambda *args,**kwargs:None
            worker._save_resume_counters(directory)
            worker.data_loader=ToyLoader(toy_sampler())
            worker._data_epoch=worker._data_iter_offset=worker.optimizer_steps=0
            worker.load_checkpoint(directory)
            self.assertEqual(worker.data_loader.sampler_epoch_at_load,3)
            self.assertEqual(list(worker.data_iter),expected)
            self.assertEqual((worker._data_epoch,worker._data_iter_offset,worker.optimizer_steps),(3,5,56))
            # Exercise the unchanged native loop's epoch transition.
            worker._data_epoch+=1
            worker.data_loader.sampler.set_epoch(worker._data_epoch)
            self.assertEqual(list(worker.data_loader),list(toy_sampler(4)))

    def test_manifest_topology_and_inconsistent_epoch_rejected_before_parent_load(self):
        class NativeBoundary:
            def load_checkpoint(self,path):
                raise AssertionError('Parent must not run for invalid resume provenance')
        Worker=worker_class_for_test(NativeBoundary)
        with tempfile.TemporaryDirectory() as directory:
            worker=Worker()
            worker._rank=0
            worker._world_size=1
            worker.micro_batch_size=1
            worker.data_config={'manifest_sha256':'correct'}
            valid={'rank':0,'world_size':1,'manifest_sha256':'correct','micro_batch_size':1,
                   'data_epoch':2,'sampler_epoch':2,'data_iter_offset':3,'optimizer_steps':4}
            for change in ({'manifest_sha256':'wrong'},{'world_size':2},{'sampler_epoch':0},
                           {'data_iter_offset':-1},{'micro_batch_size':2}):
                record={'version':1,'ranks':[{**valid,**change}]}
                (Path(directory)/'xvla_sft_data_state.json').write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    worker.load_checkpoint(directory)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable: real StatefulDataLoader NOT RUN')
class TorchLoaderResumeTests(unittest.TestCase):
    def test_real_stateful_loader_restores_later_epoch_order_and_cursor(self):
        import torch
        from torch.utils.data import DistributedSampler,TensorDataset
        from torchdata.stateful_dataloader import StatefulDataLoader
        dataset=TensorDataset(torch.arange(17))
        def build(epoch):
            sampler=EpochCursorSampler(DistributedSampler(dataset,num_replicas=1,rank=0,
                                                          shuffle=True,seed=1234))
            sampler.set_epoch(epoch)
            return StatefulDataLoader(dataset,batch_size=2,sampler=sampler,num_workers=0)
        original=build(3)
        iterator=iter(original)
        next(iterator)
        next(iterator)
        state=copy.deepcopy(original.state_dict())
        # TorchData 0.11 stores custom sampler state under its batch-iterator
        # snapshot; _index_sampler_state can legitimately be None. Verify the
        # public resume behavior rather than assuming a private nesting layout.
        expected=[batch[0].tolist() for batch in iterator]
        restored=build(0)
        restored.load_state_dict(state)
        self.assertEqual([batch[0].tolist() for batch in restored],expected)
        self.assertEqual(restored.sampler.epoch,3)


if __name__=='__main__':
    unittest.main()
