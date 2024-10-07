import Navbar from './components/Navbar'
import { BrowserRouter as Router , Routes , Route} from 'react-router-dom'

import Home from './pages/Home'
import About from './pages/About'
import Team from './pages/Team'
import Work from './pages/Work'
import Tool from './pages/Tool'

export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/work" element={<Work />} />
        <Route path="/team" element={<Team />} />
        <Route path="/tool" element={<Tool />} />
        
      </Routes>
    </Router>

    
  
  
  
  
  
  );
}

