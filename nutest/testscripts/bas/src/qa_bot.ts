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
      // Trim output to remove stray newlines/whitespace
      const trimmedOutput = output.trim();
      
      if (!trimmedOutput) {
        reject(new Error('No output from Python script'));
        return;
      }

      try {
        const response = JSON.parse(trimmedOutput);
        if (response.error) reject(new Error(response.error));
        resolve(response.result);
      } catch (e) {
        reject(new Error(
          `JSON parse failed: ${(e as Error).message}\n` +
          `Raw output: ${trimmedOutput}\n` +
          `Stderr: ${errorOutput}`
        ));
      }
    });
  });
}
