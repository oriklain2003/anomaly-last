import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DesktopApp } from './DesktopApp';
import { MobileApp } from './MobileApp';
import { ComparisonPage } from './ComparisonPage';
import { TestNewAirspacePage } from './TestNewAirspacePage';
import { DataExplorerPage } from './DataExplorerPage';
import { IntelligencePage } from './IntelligencePage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DesktopApp />} />
        <Route path="/mobile" element={<MobileApp />} />
        <Route path="/comparison" element={<ComparisonPage />} />
        <Route path="/test_new_airspace" element={<TestNewAirspacePage />} />
        <Route path="/explorer" element={<DataExplorerPage />} />
        <Route path="/intelligence" element={<IntelligencePage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
