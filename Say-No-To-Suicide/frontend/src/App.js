import React, { useState } from 'react';
import './App.css';

const highlightText = (text, words) => {
    if (!words.length) {
        return text;
    }
    // Escape special characters for regex and create a pattern
    const pattern = words.map(word => word.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')).join('|');
    const regex = new RegExp(`\\b(${pattern})\\b`, 'gi');

    const parts = text.split(regex);

    return parts.map((part, index) => {
        const isMatch = words.some(word => new RegExp(`^${word}$`, 'i').test(part));
        if (isMatch) {
            return <mark key={index}>{part}</mark>;
        }
        return part;
    });
};

function App() {
    const [text, setText] = useState('');
    const [score, setScore] = useState(null);
    const [alarmingWords, setAlarmingWords] = useState([]);
    const [analyzedText, setAnalyzedText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');
        setScore(null);
        setAlarmingWords([]);
        setAnalyzedText(text); // Store the text that was analyzed

        try {
            const response = await fetch('http://localhost:5001/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong');
            }

            const data = await response.json();
            setScore(data.score);
            setAlarmingWords(data.alarmingWords || []);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Suicidal Ideation Detector</h1>
                <p>Enter text below to analyze its suicidal ideation score.</p>
                <form onSubmit={handleSubmit}>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="I'm feeling..."
                        rows="5"
                        cols="60"
                        required
                    />
                    <br />
                    <button type="submit" disabled={isLoading}>
                        {isLoading ? 'Analyzing...' : 'Analyze'}
                    </button>
                </form>
                {score !== null && (
                    <div className="result">
                        <h2>Analysis Result</h2>
                        <p>Suicidality Score (1-10): <strong>{score.toFixed(2)}</strong></p>
                        {alarmingWords.length > 0 && (
                            <div className="highlighted-text">
                                <h3>Highlighted Text</h3>
                                <p>{highlightText(analyzedText, alarmingWords)}</p>
                            </div>
                        )}
                    </div>
                )}
                {error && (
                    <div className="error">
                        <p>Error: {error}</p>
                    </div>
                )}
            </header>
        </div>
    );
}

export default App;
