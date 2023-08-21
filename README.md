# aiira_insect_detection

```bash
roslaunch aiira_detection insect_classification.launch

# default image_topic: /theia/left/image_raw
roslaunch aiira_detection insect_classification.launch image_topic:="/my_image_topic"

# launch with additional parameters
roslaunch aiira_detection insect_classification.launch image_topic:="/my_image_topic" \
    model_file:=... \
    insect_names:=... \
    classes_file:=... \
    display_positives_only:=...

```
