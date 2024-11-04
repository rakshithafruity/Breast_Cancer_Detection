import React from "react";
import { Link } from "react-router-dom";
import styles from './Navbar.module.css'
import { useState } from "react";
import { useEffect } from "react";

export default function Navbar(){
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);

    const toggleDropdown = () => {
      setIsDropdownOpen(!isDropdownOpen);
    };
  
    const handleOutsideClick = (event) => {
      if (isDropdownOpen && !event.target.closest('.relative')) {
        setIsDropdownOpen(false);
      }
    };
  
    useEffect(() => {
      // Add event listener on component mount, remove on unmount
      document.addEventListener('click', handleOutsideClick);
      return () => document.removeEventListener('click', handleOutsideClick);
    }, [isDropdownOpen]);
  
    return (
      <nav className="bg-gray-800 absolute top-0 w-full z-50 left-0">
        <div className="flex justify-between items-center px-4 py-6 mx-auto lg:max-w-7xl md:max-w-2xl sm:max-w-full">
          <Link to="/" className="text-3xl font-bold text-white">Your Dashboard</Link>
          <div className="flex items-center space-x-4">
            <Link to="/home" className="text-white hover:bg-gray-700 px-3 py-2 rounded-md">
              Dashboard
            </Link>
            <Link to="/about" className="text-white hover:bg-gray-700 px-3 py-2 rounded-md">
              About
            </Link>
            {/* Rest of the navigation links */}
            <div className="relative">
              <button
                onClick={toggleDropdown}
                className="flex items-center focus:outline-none"
              >
                <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
                  <i className="text-white text-lg">i</i>
                </div>
              </button>
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-sm dark:bg-gray-800">
                  <ul className="space-y-1 py-2">
                    <li>
                      <Link to="/profile" className="block px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700">
                        Your Profile
                      </Link>
                    </li>
                    <li>
                      <Link to="/sign-out" className="block px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700">
                        Sign Out
                      </Link>
                    </li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      </nav>
    )
}
