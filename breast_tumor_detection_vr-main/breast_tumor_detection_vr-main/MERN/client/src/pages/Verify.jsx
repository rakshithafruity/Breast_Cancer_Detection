import React from 'react'
import { useEffect } from 'react'
import axios from 'axios'
import { Link } from 'react-router-dom'
import 'bootstrap/dist/css/bootstrap.min.css'
import styles from '../styles/verify.module.css'
import image from '../images/verifiedLogo.png'


export default function Verify() {
    const id = window.location.pathname.split('/')[2]
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.put(`/verify/${id}`)
                console.log(response)
            } catch (error) {
                console.log(error)
            }
        }
        
        fetchData();
    }, []); // empty dependency array to run only once
    
  return (
    <div className='d-flex flex-column justify-content-center align-items-center'>
      <img src={image} alt="verified logo" />
      <h1>Your Email has been Verified Successfully!</h1>
      <Link to="/login"><button type='button' className={styles.blue_btn}>Login</button></Link>
    </div>
  )
}
