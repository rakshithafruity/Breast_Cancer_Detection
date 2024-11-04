const mongoose = require('mongoose')
const {Schema} = mongoose

const userSchema = new Schema({
    name: String,
    email:{
        type: String,
        unique: true
    },
    password: String,
    phone: String,
    otp:{
        type: String,
        default: ''
    },
    verified:{
        type: Boolean,
        default: false
    }
})

const UserModel = mongoose.model('User', userSchema)

module.exports = UserModel