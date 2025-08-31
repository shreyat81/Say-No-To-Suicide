const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = 5001; // Using a different port to avoid conflicts with React dev server

app.use(cors());
app.use(express.json());

app.post('/predict', (req, res) => {
    const { text } = req.body;

    if (!text) {
        return res.status(400).json({ error: 'Text is required' });
    }

    // IMPORTANT: The path to the python executable in the virtual environment
    const pythonExecutable = 'python';
    const pythonScript = path.join(__dirname, '..', 'predict.py');
    const rootDir = path.join(__dirname, '..');

    const pythonProcess = spawn(pythonExecutable, [pythonScript, text, '--json'], { cwd: rootDir });

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
    });

    let errorOutput = '';
    pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error(`Python script exited with code ${code}`);
            console.error('Stderr:', errorOutput);
            return res.status(500).json({ error: 'Failed to run prediction model.', details: errorOutput });
        }

        try {
            // The python script now outputs a JSON string when called from this server
            const result = JSON.parse(output);
            res.json(result);
        } catch (parseError) {
            console.error('Error parsing python script JSON output:', parseError);
            console.error('Raw output:', output);
            res.status(500).json({ error: 'Failed to parse prediction output.', details: output });
        }
    });
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
