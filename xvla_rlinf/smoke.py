"""Evidence-producing checks. Real model checks require an allocated GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def identity():
    import torch,transformers,peft,rlinf
    return {'python':sys.executable,'rlinf':rlinf.__file__,'torch':torch.__version__,
            'transformers':transformers.__version__,'peft':peft.__version__,
            'job_id':os.environ.get('SLURM_JOB_ID'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['imports','alignment']);a=p.parse_args()
    from . import register
    from .sft_entry import assert_pinned_runtime
    assert_pinned_runtime();register()
    result={'identity':identity(),'mode':a.mode,'started':time.time()}
    import torch
    from rlinf.config import SupportedModel
    assert SupportedModel('xvla').value=='xvla'
    if a.mode=='alignment':
        if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
            raise RuntimeError('Alignment needs a Slurm-allocated GPU; no mocked fallback')
        import numpy as np
        from PIL import Image
        from omegaconf import OmegaConf
        from .policy import XVLAPolicy
        from .sampler import clean_action_prediction
        from .transforms import prepare_processor_inputs,env_actions_from_model
        from .vendor.reference_libero import (_flip_agentview,reference_environment_actions,
                                             SOURCE_COMMIT,SOURCE_PATH,SOURCE_SHA256)
        from unittest.mock import patch
        cfg=OmegaConf.create({'model_path':os.environ['XVLA_MODEL_PATH'],
              'add_value_head':False,'xvla':{'update_scope':'prompt_heads'}})
        model=XVLAPolicy.from_checkpoint(cfg,torch.float32).cuda().eval()
        horizon=int(model.core.num_actions)
        torch.cuda.reset_peak_memory_stats()
        rng=np.random.default_rng(2026)
        images=rng.integers(0,256,(2,256,256,3),dtype=np.uint8)
        obs={'main_images':images,'wrist_images':images[::-1].copy(),
             'task_descriptions':['put the black bowl on the plate']*2}
        adapter=prepare_processor_inputs(model.processor,obs,image_orientation='native_raw')
        reference=model.processor(images=[[Image.fromarray(_flip_agentview(images[i]).copy()),
                  Image.fromarray(obs['wrist_images'][i])] for i in range(2)],
                  language_instruction=obs['task_descriptions'])
        for key in reference:torch.testing.assert_close(adapter[key],reference[key],rtol=0,atol=0)
        assert adapter['image_mask'].tolist()==[[True,True,False]]*2
        proprio=torch.zeros((2,20),device='cuda');proprio[:,3]=1;proprio[:,7]=1
        inputs={k:v.cuda() for k,v in adapter.items()}
        inputs.update(proprio=proprio,domain_id=torch.full((2,),3,device='cuda',dtype=torch.long))
        # Observe, without changing, the official postprocessor's raw input.
        # Comparing only post-sigmoid grippers could hide a raw-logit error.
        reference_raw={}
        official_postprocess=model.core.action_space.postprocess
        def capture_native_raw(action):
            reference_raw['action']=action.detach().clone()
            return official_postprocess(action)
        torch.manual_seed(1234)
        with patch.object(model.core.action_space,'postprocess',side_effect=capture_native_raw):
            original=model.core.generate_actions(**inputs,steps=10)
        torch.manual_seed(1234);noise=torch.randn((2,horizon,20),device='cuda',dtype=proprio.dtype)
        with torch.no_grad():raw,_=clean_action_prediction(model.core,inputs,noise,steps=10)
        torch.testing.assert_close(raw,reference_raw['action'],rtol=1e-5,atol=1e-6)
        output=model.core.action_space.postprocess(raw.clone())
        torch.testing.assert_close(output,original,rtol=1e-5,atol=1e-6)
        converted=env_actions_from_model(output,rotation_backend='robosuite')
        # Independent fixed-source client conversion. Reusing the adapter on
        # both sides would fail to detect a shared rotation/gripper bug.
        native=reference_environment_actions(original.detach().cpu().numpy())
        np.testing.assert_allclose(converted,native,rtol=1e-5,atol=1e-6)
        result.update(processor_exact=True,raw_max_abs=float((raw-reference_raw['action']).abs().max()),
                      postprocess_max_abs=float((output-original).abs().max()),
                      environment_action_max_abs=float(np.max(np.abs(converted-native))),
                      environment_actions_equal=True,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                      checkpoint_chunk_length=horizon,native_steps=10,dtype=str(proprio.dtype),
                      reference_client={'source_commit':SOURCE_COMMIT,'path':SOURCE_PATH,'sha256':SOURCE_SHA256},
                      source='vendor fixed Git source; HF custom Python not executed',
                      caveat='fixed synthetic observation; real simulation is a separate acceptance item')
    result['status']='PASS';result['elapsed_seconds']=time.time()-result['started']
    folder=Path(os.environ.get('XVLA_RUN_DIR','.'));folder.mkdir(parents=True,exist_ok=True)
    (folder/(a.mode+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
