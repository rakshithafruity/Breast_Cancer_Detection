import React from 'react'
import { useState } from 'react'
import axios from 'axios'
import { toast } from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'
import styles from '../styles/register.module.css'
import { Link } from 'react-router-dom'
import { FaInfoCircle } from 'react-icons/fa';
// import {crypto} from 'crypto'


export default function Register() {
    const navigate = useNavigate()
    const [data, setData] = useState({
        name:'',
        email:'',
        password:'',
        phone:'',
    })

    const CustomToast = ({ message }) => (
    <div className="custom-toast">
        <FaInfoCircle className="info-icon" />
        <span>{message}</span>
    </div>
    );
    const registerUser = async (e) => {
        e.preventDefault()
        const {email, name, password, phone} = data
        try {
            const{data} = await axios.post('/register', {
                name, email, password, phone
            })
            if(data.error){
                toast.error(data.error)
            }
            else{
                setData({})
                toast.loading('Sending Verification Email...', {
                    duration: 1000
                })
                setTimeout(() => {
                    toast('Verification Email Sent! Please Verify.', {
                    icon: 'ℹ️',
                    iconTheme: {
                      primary: '#FFD700', // Yellow color for the icon
                      secondary: '#FFFAE5' // Light yellow for the background
                    },
                  });
            }
            , 1900)}
        } catch (error) {
            console.log(error)
        }
    }


    return (
        <div className={styles.signup_container}>
        <div className={styles.signup_form_container}>
            <div className={styles.left}>
                <h1>Already Registered?</h1>
                <Link to="/login">
                    <button type="button" className={styles.white_btn}>
                        Sign In
                    </button>
                </Link>
            </div>
            <div className={styles.right}>
                <form className={styles.form_container} onSubmit={registerUser}>
                    <h1>Create Account</h1>
                    <input
                        type="text"
                        placeholder="Name"
                        name="name"
                        onChange={(e)=> setData({...data, name:e.target.value})}
                        value={data.name}
                        required
                        className={styles.input}
                    />
                    <input
                        type="email"
                        placeholder="Email"
                        name="email"
                        onChange={(e)=> setData({...data, email:e.target.value})}
                        value={data.email}
                        required
                        className={styles.input}
                    />
                    <input
                        type="password"
                        placeholder="Password"
                        name="password"
                        onChange={(e)=> setData({...data, password:e.target.value})}
                        value={data.password}
                        required
                        className={styles.input}
                    />
                    <div>
                        <input 
                            type="tel" 
                            placeholder="Phone Number"
                            name="phone"
                            onChange={(e)=> setData({...data, phone:e.target.value})}
                            value={data.phone}
                            className={styles.input}
                        />
                    </div>
                    {/* {error && <div className={styles.error_msg}>{error}</div>} */}
                    <button type="submit" className={styles.green_btn}>
                        Sign Up
                    </button>
                </form>
            </div>
        </div>
    </div>
  )
}


