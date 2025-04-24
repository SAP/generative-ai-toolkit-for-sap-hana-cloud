// test.ts
import { callPythonStringProcessor } from './qa_bot';

// Example usage
callPythonStringProcessor('Create a dataset report for me on SHAMPOO_SALES_DATA_TBL', [])
  .then(result => console.log('Processed:', result))
  .catch(error => console.error('Error:', error));
