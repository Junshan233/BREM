from BREM.common.module import I3D_BackBone
import torch
import numpy as np


def test_inference(repeats=3, clip_frames=256):
    model = I3D_BackBone(in_channels=3)
    model.eval()
    model.cuda()
    import time

    '''
    1. 输入视频时间为 25.6s, 视频为30fps, 视频帧数为768
    2. 10fps 抽帧, 输入模型的帧数为 256
    '''
    video_time = 25.6
    fps = 30
    input_fps = 10
    num_frame = video_time * fps
    clip_frames = int(input_fps * video_time)
    
    run_times = []
    x = torch.randn([1, 3, clip_frames, 96, 96]).cuda()
    warmup_times = 2
    for i in range(repeats + warmup_times):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        run_times.append(time.time() - start)

    infer_time = np.mean(run_times[warmup_times:])
    infer_fps = num_frame * (1.0 / infer_time)
    print("inference time (ms):", infer_time * 1000)
    print("infer_fps:", int(infer_fps))
    # print(y['loc'].size(), y['conf'].size(), y['priors'].size())


if __name__ == "__main__":

    # python BREM/common/I3D_speed.py configs/thumos14.yaml
    test_inference(120, 256)
