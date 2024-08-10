

import FileUpload from './api/api';
function App() {
  return (
    <div className="flex items-center justify-center h-screen bg-yellow-50">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">Upload a file to this site</h1>
    </div>
    <div className="w-full max-w-md p-8 rounded-lg shadow-md ">
        <FileUpload />
    </div>
</div>

  );
}

export default App;
