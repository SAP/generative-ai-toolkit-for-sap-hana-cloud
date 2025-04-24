import { spawn } from 'child_process';

export async function callPythonStringProcessor(question: string, chat_history: string[]): Promise<string> {
  const pythonProcess = spawn('python', ['qa_bot.py']);

  // Prepare input
  const inputData = {
    question: question,
    chat_history: chat_history
  };
  const input = JSON.stringify(inputData);
  pythonProcess.stdin.write(input);
  pythonProcess.stdin.end();

  // Collect output
  let output = '';
  let errorOutput = '';

  pythonProcess.stdout.setEncoding('utf8');
  pythonProcess.stdout.on('data', (data) => output += data);

  pythonProcess.stderr.setEncoding('utf8');
  pythonProcess.stderr.on('data', (data) => errorOutput += data);

  return new Promise((resolve, reject) => {
    pythonProcess.on('close', (code) => {
      const trimmedError = errorOutput.trim();
      if (trimmedError) {
        reject(new Error(JSON.stringify({
          error: 'Python script execution failed',
          stderr: trimmedError,
          exitCode: code
        })));
        return;
      }

      const trimmedOutput = output.trim();
      if (!trimmedOutput) {
        reject(new Error(JSON.stringify({ error: 'No output from Python script' })));
        return;
      }

      try {
        const response = JSON.parse(trimmedOutput);
        if (response.error) {
          reject(new Error(JSON.stringify({ error: response.error })));
        } else {
          resolve(response.result);
        }
      } catch (e) {
        reject(new Error(JSON.stringify({
          error: 'JSON parse failed',
          details: {
            parseError: (e as Error).message,
            rawOutput: trimmedOutput,
            stderr: errorOutput.trim()
          }
        })));
      }
    });
  });
}