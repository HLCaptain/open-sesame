import React, {useState, useEffect} from 'react';
import {useNavigate} from 'react-router-dom';

function HomePage() {
    const [user, setUser] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const userDetails = localStorage.getItem('user');
        if (userDetails) {
            setUser(JSON.parse(userDetails));
        }
    }, []);

    const handleLogout = () => {
        localStorage.removeItem('user'); // Clear user session
        navigate('/'); // Navigate back to the Welcome page
    };

    return (
        <div className="container mx-auto mt-5 p-5 shadow-lg rounded-lg">
            <h1 className="text-2xl font-bold text-center mb-4">Welcome to Your Dashboard</h1>

            {user ? (
                    <div className="p-4">
                    <h2 className="text-xl text-gray-400">Hello, {user.name}!</h2>
                    <p>Your last login score was: <span className="font-semibold">{user.score}</span></p>
                    <button onClick={handleLogout}
                            className="btn btn-success btn-lg">
                        Logout
                    </button>


                    <div className="mt-5 bg-blue-100 rounded">
                        <h3 className="text-lg text-blue-800 font-semibold">About Your Voice Verification System</h3>
                        <p>The system uses a speaker recognition model to ensure that your voice is securely verified.
                            This model, known as ECAPA-TDNN, uses complex algorithms to analyze and compare voice
                            patterns based on the VoxCeleb dataset.</p>
                    </div>
                </div>
            ) : (
                <div>
                    <p>You are currently not logged in. Please log in to view your dashboard.</p>
                    <button onClick={() => navigate('/login')}
                            className="py-2 px-4 bg-blue-500 hover:bg-blue-700 text-white font-bold rounded-lg transition duration-300">
                        Go to Login
                    </button>
                </div>
            )}
        </div>
    );
}

export default HomePage;
