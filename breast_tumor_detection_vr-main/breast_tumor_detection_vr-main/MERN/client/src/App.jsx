import './App.css'
import {Routes, Route, useLocation} from 'react-router-dom'
import Navbar from '../src/components/Navbar'
import Home from './pages/Home'
import Login from './pages/Login'
import Register from './pages/Register'
import Verify from './pages/Verify'
import axios from 'axios'
import 'bootstrap/dist/css/bootstrap.min.css'



axios.defaults.withCredentials = true
axios.defaults.baseURL = 'http://localhost:8000'

function App() {

  const location = useLocation()

  const isHome = location.pathname === '/home'

  return (
    <>
    {(isHome) && <Navbar />}
    <Routes>
      <Route path='/' element={<Register />} />
      <Route path='/login' element={<Login />} />
      <Route path='/home' element={<Home />} />
      <Route path='/verify/:id' element={<Verify />} />
    </Routes>
    </>


  )
}

export default App

