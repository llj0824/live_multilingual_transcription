### Background ###

This application will run and be listening for conversation. Then it will transcribe the conversation into english captions. 

This is primarily intended to translate chinese conversation into english.

### TODO

1. Make application continuously running
2. Record audio nearby (maybe into chunks) and then print the transcript.


### Commands

1. Convert mp4 into mp3 file
`ffmpeg -i input.mp4 -vn -ab 128k -ar 44100 -y output.mp3`