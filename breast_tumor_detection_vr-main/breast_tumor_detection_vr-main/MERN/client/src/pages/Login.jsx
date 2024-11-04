import React from 'react'
import { useState } from 'react'
import axios from 'axios'
import { toast } from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'
import styles from '../styles/login.module.css'
import { Link } from 'react-router-dom'


export default function Login() {
    const navigate = useNavigate()
    const [data, setData] = useState({
        email:'',
        password:'',
		phone:'',
		otp: '',
		flag: false
    })
    const loginUser = async (e)=> {
         e.preventDefault()
         const {email, password} = data
        try{
            const{data} = await axios.post('/login', {
                email, password
            })
            if(data.error){
                toast.error(data.error)
            }
            else{
                setData({})
				localStorage.setItem('token', data.token)
                toast.success('Logged in successfully!')
                navigate('/home')
            }
        }
        catch(error){
            console.log(error)
        }
    }

	function generateRandomNumber() {
        // Generate a random integer between 0 and 999999
        const randomNumber = Math.floor(Math.random() * 1000000);
        // Pad the number with leading zeros to ensure it is 6 digits long
        const paddedNumber = randomNumber.toString().padStart(6, '0');
        return paddedNumber;
    }

    const generateOTP = async () => {
        // window.generate = true
		// setData(prevData => ({ ...prevData, flag: true, phone, email }));
        console.log("OTP will be generated...")
        const {email, phone} = data
		try {
			const generatedOTP = generateRandomNumber();
			const {data} = await axios.post('/sendotp', { email, phone, otp: generatedOTP });
			setData(prevData => ({ ...prevData, phone, email })); // Update the state to show OTP input field
			if(data.error){
				toast.error(data.error)
			}
			else{
				setData(prevData => ({ ...prevData, flag: true, phone, email }));
				toast.loading('Sending OTP...', {
					duration: 1000
				})
				setTimeout(() => {
					toast('OTP Sent! Please Verify.', {
					icon: 'ℹ️',
					iconTheme: {
						primary: '#FFD700', // Yellow color for the icon
						secondary: '#FFFAE5' // Light yellow for the background
					},
					});
			}
			, 1900)
			}
		} catch (error) {
			toast.error('Failed to generate OTP. Please try again.');
			console.log(error);
		}
	// setData({...data, otp: generatedOTP})

    }

	const Verify = async (e) => {
		e.preventDefault()
		const {email, phone, otp} = data
		try {
			if (!otp) {
				toast.error('Please enter OTP');
				return;
			}
			const {data} = await axios.post('/verifyotp', { email, phone, otp });
			setData(prevData => ({ ...prevData, flag: true, phone, email }));
			// toast.success('OTP verified successfully!');
			if(data.error){
				toast.error(data.error)
			}
			else{
				setData(prevData => ({ ...prevData, flag: true, phone, email}));
				toast.success('OTP verified successfully!');
			}
		} catch (error) {
			toast.error('Failed to verify OTP. Please try again.');
			console.log(error);
		}
	}
      
    return (
        <div className={styles.login_container}>
			<div className={styles.login_form_container}>
				<div className={styles.left}>
					<form className={styles.form_container} onSubmit={loginUser}>
						<h1>Login to Your Account</h1>
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
                            value={data.password}
                            onChange={(e)=> setData({...data, password:e.target.value})}
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
                            className={styles.phone}
                        />
                        <button type="button" className={styles.phone_btn} onClick={generateOTP}>Generate OTP</button>
                    </div>
                    {data.flag && (
                        <>
                          <div>
                            <input 
                                type="text" 
                                placeholder="OTP"
                                name="otp"
                                onChange={(e)=> setData({...data, otp:e.target.value})}
                                value={data.otp}
                                required
                                className={styles.phone}
                            />
                            <button type="submit" className={styles.phone_btn} onClick={Verify}>Verify</button>
                          </div>
                        </>
                    )}
						{/* {error && <div className={styles.error_msg}>{error}</div>} */}
						<button type="submit" className={styles.green_btn}>
                            Sign In
						</button>
					</form>
				</div>
				<div className={styles.right}>
					<h1>New Here ?</h1>
					<Link to="/">
						<button type="button" className={styles.white_btn}>
							Register
						</button>
					</Link>
				</div>
			</div>
		</div>
  );
};


