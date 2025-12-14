import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DesktopApp } from './DesktopApp';
import { MobileApp } from './MobileApp';
import { ComparisonPage } from './ComparisonPage';
import { TestNewAirspacePage } from './TestNewAirspacePage';
import { DataExplorerPage } from './DataExplorerPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DesktopApp />} />
        <Route path="/mobile" element={<MobileApp />} />
        <Route path="/comparison" element={<ComparisonPage />} />
        <Route path="/test_new_airspace" element={<TestNewAirspacePage />} />
        <Route path="/explorer" element={<DataExplorerPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
