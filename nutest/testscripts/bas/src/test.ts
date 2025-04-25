// test.ts
import { callPythonStringProcessor } from './qa_bot';

// Example usage
// Create a dataset report for me on SHAMPOO_SALES_DATA_TBL
// Show me the last 10 records of data from SHAMPOO_SALES_DATA_TBL
callPythonStringProcessor('Create a dataset report for me on SHAMPOO_SALES_DATA_TBL with key ID', [])
  .then(result => console.log(result))
  .catch(error => console.error(error));
