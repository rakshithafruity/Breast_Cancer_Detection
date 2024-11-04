const dotenv = require('dotenv').config();

const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;
const client = require('twilio')(accountSid, authToken);

const sendSMS = async (recipient, content) => {
	try {
		const message = await client.messages
			.create({
				body: content,
				to: `+91${recipient}`,
				from: process.env.TWILIO_PHONE_NUMBER
			})

		console.log("Message sent successfully")
		console.log(message.sid)
	}
	catch (err) {
		console.log("Error occured in sendSMS")
		console.log(err)
	}
}
// sendSMS("8106849010", "Hello Testing Twilio API")
module.exports = sendSMS
