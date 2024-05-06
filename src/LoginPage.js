import React, {useState, useEffect} from 'react';
import {useNavigate} from 'react-router-dom';
import axios from 'axios';
import config from './config';

function LoginPage() {
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [audioBlob, setAudioBlob] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        // Request permission and set up MediaRecorder
        navigator.mediaDevices.getUserMedia({audio: true})
            .then(stream => {
                const recorder = new MediaRecorder(stream);
                setMediaRecorder(recorder);

                const audioChunks = [];
                recorder.ondataavailable = (e) => {
                    audioChunks.push(e.data);
                };

                recorder.onstop = () => {
                    const blob = new Blob(audioChunks, {'type': 'audio/wav'});
                    setAudioBlob(blob);
                    // Here you would typically send the blob to your backend for processing
                    handleLogin(blob);
                };
            });
    }, []);

    const startRecording = () => {
        if (mediaRecorder) {
            setIsRecording(true);
            mediaRecorder.start();
        }
    };

    const stopRecording = () => {
        if (mediaRecorder) {
            setIsRecording(false);
            mediaRecorder.stop(); // This will trigger the onstop event
        }
    };

    // Inside handleLogin function after successful verification
    const handleLogin = async (audioBlob) => {
        const formData = new FormData();
        formData.append('file', audioBlob);

        try {
            const response = await axios.post(`${config.apiUrl}/verify`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            if (response.data.verified) {
                // Store user details in localStorage
                localStorage.setItem('user', JSON.stringify(response.data));
                navigate('/home');
            } else {
                alert('Verification failed. Please try again.');
            }
        } catch (error) {
            console.error('Verification error:', error);
            alert('An error occurred during verification.');
        }
    };


    return (
        <div className="container mx-auto mt-5 p-5 shadow-lg rounded-lg">
            <h2>Login</h2>
            <p>Please say "Open Sesame" to log in</p>
            {isRecording ? (
                <button className="btn btn-danger" onClick={stopRecording}>Stop and Verify</button>
            ) : (
                <button className="btn btn-primary" onClick={startRecording}>Start Recording</button>
            )}
        </div>
    );
}

export default LoginPage;
