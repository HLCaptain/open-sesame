import React, {useState} from 'react';
import {useNavigate} from 'react-router-dom';

function RegistrationForm() {
    const [email, setEmail] = useState('');
    const [name, setName] = useState('');
    const navigate = useNavigate();

    const handleSubmit = (e) => {
        e.preventDefault();
        // Pass name and email as state to the VoiceRecorder route
        navigate('/voice-recorder', {state: {name, email}});
    };

    return (
        <div className="container mx-auto mt-5 p-5 shadow-lg rounded-lg">
            <div className="d-flex justify-content-start mb-3">
                <button className="btn" onClick={() => navigate(-1)} aria-label="Go back">
                    <i className="bi bi-arrow-left"></i>
                </button>
            </div>
            <h2>Register</h2>
            <form onSubmit={handleSubmit}>
                <div className="mb-3">
                    <label htmlFor="emailInput" className="form-label">Email address</label>
                    <input type="email" className="form-control" id="emailInput" value={email}
                           onChange={(e) => setEmail(e.target.value)} required/>
                </div>
                <div className="mb-3">
                    <label htmlFor="nameInput" className="form-label">Name</label>
                    <input type="text" className="form-control" id="nameInput" value={name}
                           onChange={(e) => setName(e.target.value)} required/>
                </div>
                <button type="submit" className="btn btn-primary btn-lg">Submit</button>
            </form>
        </div>
    );
}

export default RegistrationForm;
