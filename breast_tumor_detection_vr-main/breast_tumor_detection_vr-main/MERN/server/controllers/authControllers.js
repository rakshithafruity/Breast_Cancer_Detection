const User = require('../models/users');
const bcrypt = require('bcryptjs');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv').config();
// const sendSMS = require('../sendOTP');
const jwt = require('jsonwebtoken');
const axios = require('axios');
const cookieParser = require('cookie-parser');
const FormData = require('form-data');

const test = (req, res) => {
    res.json('test is working');
}

const hashPassword = async (password) => {
    return new Promise((resolve, reject) => {
        bcrypt.genSalt(12, (err, salt) => {
            if(err){
                return reject(err);
            }
            bcrypt.hash(password, salt, (err, hash) => {
                if(err){
                    return reject(err);
                }
                return resolve(hash);
            })
        })
    })
}

const registerUser = async (req, res) => {
    try {
        const {name, email, password, phone} = req.body;

        //Check if password meets the requirements
        const hasUpper = /[A-Z]/.test(password);
        const hasLower = /[a-z]/.test(password);
        const hasNumber = /\d/.test(password);
        const hasSpecial = /[^a-zA-Z\d]/.test(password);
        const isLongEnough = password.length >= 6;

        if (!hasUpper) {
            return res.json({ error: 'Password must contain at least one uppercase letter' });
        }
        if (!hasLower) {
            return res.json({ error: 'Password must contain at least one lowercase letter' });
        }
        if (!hasNumber) {
            return res.json({ error: 'Password must contain at least one number' });
        }
        if (!hasSpecial) {
            return res.json({ error: 'Password must contain at least one special character' });
        }
        if (!isLongEnough) {
            return res.json({ error: 'Password must be at least 6 characters long' });
        }
        const regex = /\b\d{10}\b/;
        if(!regex.test(phone)) {
            return res.json({ error: 'Please enter valid phone number'});
        }


        //Check if email entered
        const exist = await User.findOne({email});
        if(exist){
            return res.json({
                error: 'Email already exist'
            })
        };
        const hashedPassword = await hashPassword(password);
        const user = await User.create({
            name,
            email,
            password: hashedPassword,
            phone
        })
            

        const transporter = nodemailer.createTransport({
        service: 'gmail',
        host: 'smtp.gmail.com',
        port: 587,
        secure: false,
        auth: {
            user: process.env.MAIL,
            pass: process.env.MAIL_PASS,
        }
        });

        const mailOptions = {
        from:{
            name: 'UDAAN VR-LLM',
            address: process.env.MAIL
        },
        to: email,
        subject: 'Verification of Email for VR-LLM',
        text: `Please click the link below to verify your email address:\nhttp://localhost:5173/verify/${user._id}`
        };

        transporter.sendMail(mailOptions, function(error, info){
        if (error) {
            console.log(error);
        } else {
            console.log('Email sent: ' + info.response);
        }
        });

        return res.json(User)
    } catch (error) {
        console.log(error);}
}

const verified = async (req, res) => {
    const id  = req.params.id;
    const user  = await User.findByIdAndUpdate({_id:id}, {verified: true})
    .then(user => res.json(user))
    .catch(err => res.status(400).json('Error: ' + err));
}

// const sendOTP = async (req, res) => {
//     const {email, phone, otp} = req.body;
//     const user = await User.findOne({email});

//     console.log('request sent')
//     try{
//         if(!user){
//             return res.json({
//                 error: 'User does not exist'
//             })
//         }
//         if(phone != user.phone){
//             return res.json({
//                 error: 'Incorrect phone number'
//             })
//         }
//         user.otp = otp;
//         await user.save();
//         sendSMS(user.phone, "Your verification code is:\n"+otp);
//         console.log("OTP sent")
//         return res.json(User)
//     }
//     catch (error) {
//         console.log(error)}
// }

// const verifyOTP = async (req, res) => {
//     const {email, phone,otp} = req.body;
//     const user = await User.findOne({email});
//     try{
//         if(!user){
//             return res.json({
//                 error: 'User does not exist'
//             })
//         }
//         if(otp == user.otp){
//             await User.updateOne({ email: email }, { $unset: { otp: 1 } })
//             console.log("Verified")
//             return res.json({
//                 success: 'Verified'
//             })
//         }
//         else{
//             return res.json({
//                 error: 'Incorrect OTP'
//             })
//         }
//     }
//     catch (error) {
//         console.log(error)}
// }

const loginUser = async (req, res) => {
    try {
        const {email, password} = req.body; 
        const user = await User.findOne({email});

        if(!user){
            return res.json({
                error: 'User does not exist'
            })
        }

        const isVerified = user.verified;
        if(!isVerified){
            return res.json({
                error: 'Please verify your email'
            })
        }

        const comparedPassword = await bcrypt.compare(password, user.password);
        if(!comparedPassword){
            return res.json({
                error: 'Incorrect password'
            })
        }
        else{
            res.clearCookie('token');
            const token = jwt.sign(
                {userID: user._id},
                process.env.JWT_SECRET,
                {expiresIn: '1d'}
            )
            res.cookie('token', token, {httpOnly: true});
            return res.json(User)
        }
    }
    catch (error) {
        console.log(error)}
}

const uploadFileClass = async (req, res, model, input) => {
    if (res.statusCode === 401) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    if (!req.file) {
        return res.status(400).json({ message: 'No file uploaded' });
    }
    try {
        const formData = new FormData();
        formData.append(`${input}`, req.file.buffer, req.file.originalname);
        const flaskResponse = await axios.post(`http://127.0.0.1:5000/predict_${input}_${model}`, formData, {
            headers: {
                ...formData.getHeaders()
            },
            responseType: `${input==='video' ? 'stream' : 'arraybuffer'}`
        });
        if(input==='video') flaskResponse.data.pipe(res);
        else {
            res.set('Content-Type', 'image/png'); // Set the response content type to image
            res.send(Buffer.from(flaskResponse.data, 'binary')); // Send the response as binary data
        }
    } catch (error) {
        console.log(error);
        res.status(500).json({ message: 'Error processing video', error: error.message });
    }
};

const uploadFileSeg = async (req, res, model, input) =>{
    // console.log('hi')
    if (res.statusCode === 401) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    if (!req.files || !req.files['image'] || !req.files['mask']) {
        return res.status(400).json({ message: 'No files uploaded' });
    }
    try {
        const formData = new FormData();
        formData.append('image', req.files['image'][0].buffer, req.files['image'][0].originalname);
        formData.append('mask', req.files['mask'][0].buffer, req.files['mask'][0].originalname);

        const flaskResponse = await axios.post(`http://127.0.0.1:5000/predict_${input}`, formData, {
            headers: {
                ...formData.getHeaders()
            },
            responseType: `${input==='video' ? 'stream' : 'arraybuffer'}`
        });
        if(input==='video') flaskResponse.data.pipe(res);
        else {
            res.set('Content-Type', 'image/png'); // Set the response content type to image
            res.send(Buffer.from(flaskResponse.data, 'binary')); // Send the response as binary data
        }

    } catch (error) {
        console.log(error);
        res.status(500).json({ message: 'Error processing video', error: error.message });
    }
}


module.exports = {
    test,
    registerUser,
    loginUser,
    verified,
    uploadFileClass,
    uploadFileSeg,
    // sendOTP,
    // verifyOTP,
}