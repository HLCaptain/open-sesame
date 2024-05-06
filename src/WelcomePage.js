import React from 'react';
import {useNavigate} from 'react-router-dom';

function WelcomePage() {
    const navigate = useNavigate();

    return (
        <div className="container mx-auto mt-5 p-5 shadow-lg rounded-lg">
            <h2 className="mb-4">Welcome to the "Open Sesame" UX Laboratory Tutorial</h2>
            <p className="mb-4">
                This tutorial is part of the <strong>UX Laboratory</strong> course offered by the Faculty of Electrical
                Engineering and Informatics at the Budapest University of Technology and Economics. The course is
                designed to equip students with practical knowledge and skills in UX design, integrating machine
                learning models into user interfaces, and conducting system evaluations and testing.
            </p>
            <p className="mb-4">
                In this specific tutorial, we will focus on the integration of automatic speaker recognition
                technologies into user interfaces. You will learn how to use the <strong>ECAPA-TDNN</strong> model
                from <strong>SpeechBrain</strong>, trained on the Voxceleb dataset, to implement speaker verification
                functionalities within a web application.
            </p>
            <p className="mb-4">
                By the end of this tutorial, you will have hands-on experience with creating user interfaces that
                leverage state-of-the-art machine learning models for speaker verification, following best practices in
                UX design.
            </p>
            <div className="d-grid gap-2 col-6 mx-auto">
                <button onClick={() => navigate('/register')} className="btn btn-primary btn-lg">Register</button>
                <button onClick={() => navigate('/login')} className="btn btn-secondary btn-lg">Log In</button>
            </div>
            <p className="mt-4">
                <small>
                    For more information about the course and other tutorials, please visit the <a
                    href="http://smartlab.tmit.bme.hu/UX" target="_blank" rel="noopener noreferrer">course website</a>.
                </small>
            </p>
        </div>
    );
}

export default WelcomePage;
