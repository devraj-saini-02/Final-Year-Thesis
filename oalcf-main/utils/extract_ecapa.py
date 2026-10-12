import os
import glob
import soundfile as sf
import resampy
import numpy as np
import torch
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier

from device import get_device

from pdb import set_trace

def main():
    device = get_device()
    # model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
    #                                     savedir="pretrained_models/spkrec-xvect-voxceleb",
    #                                     run_opts={"device":"cuda"})
    model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                        savedir="pretrained_models/lang-id-voxlingua107-ecapa",
                                        run_opts={"device": str(device)})

    fs = 16000
    subdirs = ['train', 'dev', 'test']
    for subdir in subdirs:
        in_files = glob.glob(os.path.join(subdir, '*.mp3'))
        # out_dir = os.path.join('xvectors', subdir)
        out_dir = os.path.join('ecapalang', subdir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        print(subdir)
        n_failed = 0
        for ff in tqdm(in_files):
            out_file = os.path.join(out_dir, os.path.basename(ff).replace('.mp3', '.npy'))
            if os.path.exists(out_file):
                continue
            try:
                sig,orig_fs = sf.read(ff)
                if orig_fs != fs:
                    sig = resampy.resample(sig, orig_fs, fs)
                with torch.no_grad():
                    output = model.encode_batch(torch.from_numpy(sig).float().unsqueeze(0).to(device))
                output = torch.mean(output[0], dim=0).cpu().numpy()
                np.save(out_file, output)
            except Exception as e:
                n_failed += 1
                print(f'FAILED to decode {ff}: {e}')
                continue
        print(f'{subdir}: {n_failed}/{len(in_files)} clips failed to decode')


if __name__=='__main__':
    main()