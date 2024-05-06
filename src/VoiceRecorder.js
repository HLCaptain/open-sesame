import React, {useState, useEffect} from 'react';
import axios from 'axios';
import { useLocation, useNavigate } from 'react-router-dom';
import config from './config';

function VoiceRecorder() {
    const location = useLocation();
    const { name, email } = location.state || { name: '', email: '' };
    const [recording, setRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [audioURLs, setAudioURLs] = useState([]);
    const [audioBlobs, setAudioBlobs] = useState([]);
    const [currentRecordingIndex, setCurrentRecordingIndex] = useState(0);
    const [seconds, setSeconds] = useState(0); // State to keep track of recording duration

    const navigate = useNavigate();
    const sentences = [
        "Please say 'Open Sesame'"
    ];

    useEffect(() => {
        navigator.mediaDevices.getUserMedia({audio: true})
            .then(stream => {
                const recorder = new MediaRecorder(stream);
                setMediaRecorder(recorder);

                recorder.ondataavailable = (e) => {
                    setAudioBlobs(prev => [...prev, e.data]);
                    const audioURL = URL.createObjectURL(e.data);
                    setAudioURLs(prev => [...prev, audioURL]);
                    setCurrentRecordingIndex(prev => prev + 1);
                };
            });
    }, []);

    useEffect(() => {
        let interval = null;
        if (recording) {
            interval = setInterval(() => {
                setSeconds(prevSeconds => prevSeconds + 1);
            }, 100);
        } else {
            clearInterval(interval);
        }
        return () => clearInterval(interval); // Cleanup interval on component unmount
    }, [recording]);

    const startRecording = () => {
        setSeconds(0); // Reset seconds to 0 before starting a new recording
        mediaRecorder.start();
        setRecording(true);
    };

    const stopRecording = () => {
        mediaRecorder.stop();
        setRecording(false);
    };

    const uploadAudio = async () => {
        const formData = new FormData();
        audioBlobs.forEach((blob, index) => {
            formData.append(`file${index + 1}`, blob); // Adjusted to match file naming convention
        });
        // Append user name and email to formData
        formData.append('name', name);
        formData.append('email', email);

        try {
            await axios.post(`${config.apiUrl}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            alert('Upload successful');
            navigate('/login', {state: {name: name, email: email}});
        } catch (error) {
            console.error('Upload failed:', error);
        }
    };


    return (
        <div className="container mx-auto mt-5 p-5 shadow-lg rounded-lg">
            <div className="d-flex justify-content-start mb-3">
                <button className="btn" onClick={() => navigate(-1)} aria-label="Go back">
                    <i className="bi bi-arrow-left" style={{fontSize: '24px'}}></i>
                </button>
            </div>

            <h2 className="text-center">Record Your Voice</h2>
            {currentRecordingIndex < sentences.length && (
                <>
                    <p className="mt-3 text-center">Please follow the instructions below:</p>
                    <p className="text-muted text-center">"{sentences[currentRecordingIndex]}"</p>
                    <div className="d-grid gap-2">
                        {recording && <div className="recording-indicator mx-auto mb-2"></div>}
                        {recording && seconds > 0 &&
                            <div className="text-center mb-2">Recording Time: {seconds} seconds</div>}
                        <button className="btn btn-danger btn-lg" onClick={startRecording} disabled={recording}>Start
                            Recording
                        </button>
                        <button className="btn btn-secondary btn-lg" onClick={stopRecording} disabled={!recording}>Stop
                            Recording
                        </button>
                    </div>
                </>
            )}
            {currentRecordingIndex >= sentences.length && (
                <div className="d-grid gap-2">
                    <button className="btn btn-success btn-lg" onClick={uploadAudio}>Upload</button>
                </div>
            )}
            <div>
                {audioURLs.map((url, index) => (
                    <audio key={index} controls src={url} className="w-100 mt-2"></audio>
                ))}
            </div>
        </div>

    );
}

export default VoiceRecorder;
