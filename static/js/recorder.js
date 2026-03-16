
// Audio Recorder Utility

const AudioRecorder = {
    mediaRecorder: null,
    audioChunks: [],
    stream: null,
    isRecording: false,

    // Start recording
    async start() {
        if (this.isRecording) return;

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(this.stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = event => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            return true;
        } catch (err) {
            console.error("Error starting recording:", err);
            throw err;
        }
    },

    // Stop recording and return blob
    stop() {
        return new Promise((resolve, reject) => {
            if (!this.isRecording || !this.mediaRecorder) {
                resolve(null);
                return;
            }

            this.mediaRecorder.onstop = () => {
                const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
                const audioBlob = new Blob(this.audioChunks, { type: mimeType });
                this.cleanup();
                resolve(audioBlob);
            };

            this.mediaRecorder.stop();
            this.isRecording = false;
        });
    },

    // Cleanup stream tracks
    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.mediaRecorder = null;
        this.audioChunks = [];
    },

    // Get current recording state
    getState() {
        return this.isRecording;
    }
};

window.AudioRecorder = AudioRecorder;
