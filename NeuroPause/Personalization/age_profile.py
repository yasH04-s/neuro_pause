# age_profile.py

"""
This file defines user age profiles and their associated nudge strategies.

Expected input: 
- User data containing age information.

Expected output: 
- Age-based profiles that can be used to determine appropriate nudges for users.
"""

class AgeProfile:
    def __init__(self, age):
        self.age = age
        self.profile = self.define_profile()

    def define_profile(self):
        """
        Define user profile based on age.
        
        Returns:
            dict: A dictionary containing age profile and associated nudge strategies.
        """
        if self.age < 18:
            return {
                "profile": "Teenager",
                "nudge_strategy": "Engagement through gamification"
            }
        elif 18 <= self.age < 35:
            return {
                "profile": "Young Adult",
                "nudge_strategy": "Social sharing and competition"
            }
        elif 35 <= self.age < 60:
            return {
                "profile": "Adult",
                "nudge_strategy": "Goal setting and reminders"
            }
        else:
            return {
                "profile": "Senior",
                "nudge_strategy": "Health-focused nudges"
            }

    def get_profile(self):
        """
        Get the user profile and nudge strategy.

        Returns:
            dict: The user's age profile and nudge strategy.
        """
        return self.profile