// test.ts
import { callPythonStringProcessor } from './qa_bot';

// Example usage
callPythonStringProcessor('Show me all the trained models', ['First chat history is here', 'Second chat history is here'])
  .then(result => console.log('Processed:', result))
  .catch(err => console.error('Error:', err));
