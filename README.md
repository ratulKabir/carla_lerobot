## Carla dataset in LeRobot Datset format to finetune smolvla on it


### Generating a Video from Images

To generate a video from a sequence of images, use the following `ffmpeg` command:

```bash
ffmpeg -framerate 4 -i /home/ratul/Workstation/ratul/simlingo/database/simlingo_1_data/data/simlingo/validation_1_scenario/routes_validation/random_weather_seed_2_balanced_150/Town13_Rep0_10_route0_01_11_13_24_48/rgb/%04d.jpg -c:v libx264 -pix_fmt yuv420p /home/ratul/Workstation/ratul/database/lerobot/carla_lerobot/videos/chunk-000/observation.images.main/episode_000000.mp4
```

This command creates a video at 4 frames per second from the images in the specified directory.