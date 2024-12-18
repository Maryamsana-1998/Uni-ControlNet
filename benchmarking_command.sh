python3 -m compressai.utils.video.eval_model pretrained ../test_uvg ../test_uvg -a ssf2020 -q 1 -o results.json

ffmpeg -y -f rawvideo \
  -pixel_format yuv420p \
  -s 1920x1080 \
  -framerate 120 \
  -i Beauty_1920x1080_120fps_420_8bit_YUV.yuv \
  -c:v libx264 -crf 30 -preset medium \
  Beauty_encoded_low_bitrate.mp4

ffmpeg -i Beauty_encoded_low_bitrate.mp4 \
  -f rawvideo -pixel_format yuv420p -s 1920x1080 -framerate 120 \
  -i Beauty_1920x1080_120fps_420_8bit_YUV.yuv \
  -filter_complex "psnr=stats_file=psnr_stats.log" \
  -f null -

ffmpeg -i Beauty_encoded_low_bitrate.mp4 \
  -f rawvideo -pixel_format yuv420p -s 1920x1080 -framerate 120 \
  -i Beauty_1920x1080_120fps_420_8bit_YUV.yuv \
  -filter_complex "ssim=stats_file=ssim_stats.log" \
  -f null -
