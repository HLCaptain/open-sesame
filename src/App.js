import React from 'react';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import WelcomePage from './WelcomePage';
import RegistrationForm from './RegistrationForm';
import LoginPage from './LoginPage';
import HomePage from './HomePage';
import VoiceRecorder from './VoiceRecorder';

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<WelcomePage/>}/>
                <Route path="/register" element={<RegistrationForm/>}/>
                <Route path="/login" element={<LoginPage/>}/>
                <Route path="/home" element={<HomePage/>}/>
                <Route path="/voice-recorder" element={<VoiceRecorder/>}/>
            </Routes>
        </Router>
    );
}

export default App;
