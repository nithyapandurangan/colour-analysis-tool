import Navbar from './components/Navbar'
import { BrowserRouter as Router , Routes , Route} from 'react-router-dom'

import Home from './pages/Home'
import About from './pages/About'
import Team from './pages/Team'
import Tool from './pages/Tool'
import SkinCare from './pages/SkinCare'

export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/team" element={<Team />} />
        <Route path="/tool" element={<Tool />} />
        <Route path="/SkinCare" element={<SkinCare />} />
        
      </Routes>
    </Router>

  );
}

