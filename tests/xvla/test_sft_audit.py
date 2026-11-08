"""Canonical-state audit regression, independent of live FSDP name traversal.

Array-backed tensors test byte hashing/classification only. They do not claim
real model, FSDP, optimizer or checkpoint execution.
"""
import ast
import hashlib
from pathlib import Path
import types
import unittest

import numpy as np


class ArrayTensor:
    def __init__(self,value):
        self.value=np.asarray(value)
        self.shape=self.value.shape
        self.dtype=self.value.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return ArrayTensor(np.ascontiguousarray(self.value))

    def reshape(self,*shape):
        return ArrayTensor(self.value.reshape(*shape))

    def view(self,dtype):
        return ArrayTensor(self.value.view(dtype))

    def numpy(self):
        return self.value


class ParameterNames:
    def __init__(self,names):
        self.names=names
        self.calls=[]

    def named_parameters(self,remove_duplicate=True):
        self.calls.append(remove_duplicate)
        return iter((name,types.SimpleNamespace(requires_grad=trainable))
                    for name,trainable in self.names.items())


def audit_worker(names,state):
    source=Path(__file__).resolve().parents[2]/'xvla_rlinf/sft_entry.py'
    worker=next(node for node in ast.parse(source.read_text()).body
                if isinstance(node,ast.ClassDef) and node.name=='XVLASftWorker')
    worker.bases=[]
    worker.body=[node for node in worker.body if isinstance(node,ast.FunctionDef)
                 and node.name=='_model_fingerprints']
    namespace={'hashlib':hashlib,'torch':types.SimpleNamespace(
        uint8=np.uint8,is_tensor=lambda value:isinstance(value,ArrayTensor))}
    exec(compile(ast.Module(body=[worker],type_ignores=[]),str(source),'exec'),namespace)
    instance=namespace['XVLASftWorker']()
    instance.model=ParameterNames(names)
    instance.test_state={key:ArrayTensor(value) for key,value in state.items()}
    calls=[]
    def native_state(**kwargs):
        calls.append(kwargs)
        return instance.test_state
    instance.get_model_state_dict=native_state
    instance.state_calls=calls
    return instance


class CanonicalStateAuditTests(unittest.TestCase):
    def test_frozen_parameter_disappearing_from_live_traversal_remains_checked(self):
        worker=audit_worker({'_fsdp_wrapped_module.core.vlm.image_projection':False,
                             '_fsdp_wrapped_module.core.head':True},
                            {'core.vlm.image_projection':np.eye(2,dtype=np.float32),
                             'core.head':np.zeros((2,),np.float32),
                             'core.domain.frozen_weight':np.ones((2,),np.float32)})
        before=worker._model_fingerprints()
        # Model the observed FSDP traversal change after a real forward.
        worker.model.names={'_fsdp_wrapped_module.core.head':True}
        worker.test_state['core.head'].value[0]=.125
        after=worker._model_fingerprints()
        changed=[key for key in before if before[key]['sha256']!=after[key]['sha256']]
        self.assertEqual(changed,['core.head'])
        self.assertFalse(after['core.vlm.image_projection']['trainable'])
        self.assertEqual(worker.model.calls,[False])
        self.assertTrue(all(call=={'cpu_offload':True,'full_state_dict':True}
                            for call in worker.state_calls))
        expected=hashlib.sha256(np.eye(2,dtype=np.float32).reshape(-1).view(np.uint8).tobytes()).hexdigest()
        self.assertEqual(after['core.vlm.image_projection']['sha256'],expected)

    def test_all_tied_trainable_aliases_are_trainable_and_prefixes_are_canonical(self):
        aliases={'_fsdp_wrapped_module.core.shared.weight':True,
                 '_fsdp_wrapped_module.core.encoder.embed_tokens.weight':True}
        weights={name:np.ones((2,),np.float32) for name in aliases}
        worker=audit_worker(aliases,weights)
        before=worker._model_fingerprints()
        self.assertEqual(set(before),{'core.shared.weight','core.encoder.embed_tokens.weight'})
        self.assertTrue(all(record['trainable'] for record in before.values()))
        for tensor in worker.test_state.values():
            tensor.value[0]=2.
        after=worker._model_fingerprints()
        changed_frozen=[name for name in before if before[name]['sha256']!=after[name]['sha256']
                        and not before[name]['trainable']]
        self.assertEqual(changed_frozen,[])

    def test_missing_added_or_duplicate_state_keys_fail_instead_of_disappearing(self):
        for change in ('missing','added','duplicate'):
            worker=audit_worker({'core.head':True},
                                {'core.head':np.zeros(1,np.float32),'core.frozen':np.zeros(1,np.float32)})
            worker._model_fingerprints()
            if change=='missing':
                del worker.test_state['core.frozen']
            elif change=='added':
                worker.test_state['new.buffer']=ArrayTensor(np.zeros(1))
            else:
                worker.test_state['_fsdp_wrapped_module.core.head']=ArrayTensor(np.zeros(1))
            with self.assertRaises(RuntimeError):
                worker._model_fingerprints()

    def test_initial_trainable_alias_missing_from_state_fails(self):
        worker=audit_worker({'core.head':True,'core.alias':True},
                            {'core.head':np.zeros(1,np.float32)})
        with self.assertRaisesRegex(RuntimeError,'trainable aliases absent'):
            worker._model_fingerprints()

    def test_frozen_domain_buffer_change_is_detected(self):
        worker=audit_worker({'core.head':True},
                            {'core.head':np.zeros(1,np.float32),'core.rows.frozen_weight':np.ones(3,np.float32)})
        before=worker._model_fingerprints()
        worker.test_state['core.rows.frozen_weight'].value[2]=3.
        after=worker._model_fingerprints()
        changed_frozen=[name for name in before if before[name]['sha256']!=after[name]['sha256']
                        and not before[name]['trainable']]
        self.assertEqual(changed_frozen,['core.rows.frozen_weight'])


if __name__=='__main__':
    unittest.main()
