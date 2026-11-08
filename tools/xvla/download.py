"""Download fixed weights/assets only; HF custom Python is never executed."""
import hashlib,json,os,sys
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download

SPECS={'libero':('2toINF/X-VLA-Libero','129e71460678b7236cee6fc9707f09d9fa0c3590'),
       'pt':('2toINF/X-VLA-Pt','c1c4a64a7e03ac5b95c468bf1578f3d03651b53b')}
def main():
    root=Path(sys.argv[1]); name=sys.argv[2]; repo,revision=SPECS[name]
    dest=root/(repo.split('/')[-1]+'--'+revision[:12]);dest.mkdir(parents=True,exist_ok=True)
    errors=[]
    for endpoint in ['https://huggingface.co','https://hf-mirror.com']:
        try:
            info=HfApi(endpoint=endpoint).model_info(repo,revision=revision,files_metadata=True)
            assert info.sha==revision
            records=[]
            for entry in info.siblings:
                f=entry.rfilename
                if not f.endswith(('.json','.safetensors','.bin','.model','.txt','.merges','.vocab')) or f.endswith('.py'): continue
                path=Path(hf_hub_download(repo,f,revision=revision,local_dir=dest,endpoint=endpoint))
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
                actual=digest.hexdigest()
                lfs=getattr(entry,'lfs',None)
                expected=getattr(lfs,'sha256',None) if lfs else None
                if expected and actual!=expected: raise ValueError('Hash mismatch: '+f)
                records.append({'file':f,'bytes':path.stat().st_size,'sha256':actual,'lfs_sha256':expected})
            (dest/'verified.json').write_text(json.dumps({'repo_id':repo,'revision':revision,'endpoint':endpoint,'files':records},indent=2))
            print(json.dumps({'verified':str(dest),'files':len(records)}),flush=True);return
        except Exception as e: errors.append(endpoint+': '+str(e));print(errors[-1],flush=True)
    raise RuntimeError('; '.join(errors))
if __name__=='__main__':main()
