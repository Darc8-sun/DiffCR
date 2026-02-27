# Towards Efficient Low-rate Image Compression with Frequency-aware Diffusion Prior Refinement 
> Yichong Xia, Yimin Zhou, Jinpeng Wang, Bin Chen<br>
> :sunglasses: This work is accepted by AAAI-26
>
The overall pipeline of DiffCR
<p align="center">
    <img src="./asset/pipeline.png" style="border-radius: 0px"><br>
</p>

## <a name="cite"></a>:microscope: Performance
### Quantitative Performance on Kodak </summary>
<p align="center">
    <img src="asset/kodak_rd.png" style="border-radius: 0px"><br>
</p>

###  Qualitative Performance; Visualization </summary>
<div align="center">
    <table style="width: 80%; border-collapse: collapse;">
        <tr>
            <td align="center" style="width: 50%;">
                <a href="https://imgsli.com/NDUyMzMw">
                    <img src="asset/anime.png" style="border-radius: 0px; width: 80%; height: auto;"><br>
                </a>
            </td>
            <td align="center" style="width: 50%;">
                <a href="https://imgsli.com/NDUyMzI0">
                    <img src="asset/animal.png" style="border-radius: 0px; width: 80%; height: auto;"><br>
                </a>
            </td>
        </tr>
        <tr>
            <td align="center">
                <a href="https://imgsli.com/NDUyMzI3">
                    <img src="asset/humanface.png" style="border-radius: 0px; width: 80%; height: auto;"><br>
                </a>
            </td>
            <td align="center">
                <a href="https://imgsli.com/NDUyMzIx">
                    <img src="asset/nature.png" style="border-radius: 0px; width: 80%; height: auto;"><br>
                </a>
            </td>
        </tr>
    </table>
</div>

## <a name="cite"></a>:wrench: Preparation

1. Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) into `./ckpt/SD21`.
   ```
   wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
   ```
2. Download pretrained CLIP model (https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) into `./ckpt/SD21`.

3. Download pretrained DiffCR into `./ckpt/CR_model`.

## <a name="cite"></a>:scroll: Citation


Please cite us if our work is useful for your research.
```
@article{xia2026towards,
  title={Towards Efficient Low-rate Image Compression with Frequency-aware Diffusion Prior Refinement},
  author={Xia, Yichong and Zhou, Yimin and Wang, Jinpeng and Chen, Bin},
  journal={arXiv preprint arXiv:2601.10373},
  year={2026}
}
```

