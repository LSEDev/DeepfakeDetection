Contains all code regarding our pre-processing pipeline (including resturcturing data to be in the appropriate format for .flow_from_directory method).

The stages of pre-processing pipeline are as follows:

1. Download FaceForensics++ videos
2. Extract every fifteenth frame from each video
3. Extract one face from each frame, saving a cropped square as a .png image

Additionally, resturcturing code is provided to reorganise data for the appropriate reading-in approach.
