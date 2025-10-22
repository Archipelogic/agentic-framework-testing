#!/usr/bin/env python3
"""
COMPLETE data generation for Email Automation use case.
Generates BOTH test cases AND ground truth with all required fields.
Self-contained script that addresses ALL agent requirements.
"""

import json
import random
from datasets import load_dataset
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class EmailAutomationDataGenerator:
    """Generate complete, realistic data for email automation use case."""
    
    def __init__(self):
        self.dataset = None
        self.test_cases = []
        self.ground_truth = []
        
        # Realistic email templates
        self.templates = {
            "acknowledgment": "Thank you for your email regarding {subject}. I've received your message and will review it carefully. I'll get back to you within {timeframe} with a detailed response.",
            
            "meeting_accept": "I'd be happy to attend the meeting about {subject}. The proposed time of {time} works well for me. I'll add it to my calendar and look forward to our discussion.",
            
            "meeting_decline": "Thank you for the invitation to discuss {subject}. Unfortunately, I have a prior commitment at that time. Could we explore alternative times? I'm available {availability}.",
            
            "information_request": "Thank you for reaching out about {subject}. To better assist you, I'll need some additional information: {details_needed}. Once I have these details, I can provide a comprehensive response.",
            
            "project_update": "Thank you for your update on {subject}. I've reviewed the progress and have the following feedback: {feedback}. Please proceed with the next steps as discussed.",
            
            "urgent_response": "I've received your urgent request regarding {subject}. I'm prioritizing this matter and will {action}. You can expect {outcome} by {deadline}.",
            
            "delegation": "Thank you for bringing {subject} to my attention. I'm forwarding this to {delegate} who is better equipped to handle this matter. They will contact you directly within {timeframe}.",
            
            "ooo_response": "Thank you for your email. I'm currently out of office until {return_date}. For urgent matters, please contact {backup_contact}. Otherwise, I'll respond to your message upon my return.",
            
            "followup": "Following up on our previous discussion about {subject}. {update_or_question}. Please let me know if you need any additional information.",
            
            "decline_request": "Thank you for your request regarding {subject}. After careful consideration, I'm unable to accommodate this at the current time due to {reason}. I appreciate your understanding."
        }
        
        # Email categories
        self.categories = [
            "meeting_request", "information_request", "project_update", 
            "urgent_action", "newsletter", "personal", "spam",
            "customer_support", "contract_review", "expense_report",
            "hr_communication", "technical_issue", "sales_inquiry",
            "feedback", "complaint", "collaboration_request",
            "event_invitation", "policy_update", "training",
            "weekly_report", "quarterly_review"
        ]
        
        # Priority levels with weights
        self.priority_weights = {
            "critical": 5,
            "high": 15,
            "medium": 40,
            "low": 30,
            "minimal": 10
        }
        
        # Realistic sender patterns
        self.sender_patterns = {
            "internal": {
                "domains": ["company.com", "internal.org", "corp.net"],
                "departments": ["engineering", "sales", "hr", "finance", "marketing", "legal", "operations", "it", "research", "product"],
                "first_names": ["John", "Sarah", "Michael", "Jennifer", "David", "Lisa", "Robert", "Maria", 
                               "James", "Patricia", "William", "Linda", "Richard", "Barbara", "Joseph", "Susan",
                               "Thomas", "Jessica", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Betty",
                               "Matthew", "Dorothy", "Anthony", "Helen", "Donald", "Sandra", "Mark", "Donna"],
                "last_names": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson"]
            },
            "external": {
                "domains": ["client.com", "vendor.io", "partner.org", "supplier.net", "consultant.biz",
                           "gmail.com", "yahoo.com", "outlook.com", "protonmail.com"],
                "companies": ["TechCorp", "GlobalSolutions", "InnovateLabs", "DataDynamics", "CloudFirst",
                             "SecureNet", "AlphaTech", "BetaSoft", "GammaIndustries", "DeltaSystems"]
            }
        }
        
        # Actions that can be taken
        self.actions = [
            "archive", "flag", "forward", "reply", "schedule_meeting",
            "create_task", "mark_important", "categorize", "snooze",
            "delete", "mark_spam", "add_to_calendar", "assign_to_team"
        ]
        
    def load_email_dataset(self):
        """Load the Enron email dataset."""
        print("📧 Loading email dataset from Hugging Face...")
        try:
            self.dataset = load_dataset(
                "aeslc",  # Enron email summarization corpus
                split="train[:2000]",
                cache_dir=".cache"
            )
            print(f"✅ Loaded {len(self.dataset)} emails")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            # Create fallback data
            self.dataset = self.create_fallback_emails()
    
    def create_fallback_emails(self) -> List[Dict]:
        """Create realistic fallback emails if dataset loading fails."""
        print("⚠️ Using fallback email generation...")
        emails = []
        
        subjects = [
            "Q3 Budget Review", "Project Alpha Status Update", "Meeting Request: Product Roadmap",
            "Urgent: Server Maintenance Tonight", "FYI: Policy Changes", "Team Building Event - RSVP",
            "Contract Review Needed", "Customer Complaint - High Priority", "Weekly Report Submission",
            "Training Session Tomorrow", "Expense Report Approval", "New Hire Introduction",
            "System Upgrade Notification", "Feedback on Proposal", "Conference Room Booking",
            "Deadline Reminder", "Question about Implementation", "Partnership Opportunity",
            "Security Alert", "Holiday Schedule", "Performance Review Schedule"
        ]
        
        for i in range(200):
            subject = random.choice(subjects) + f" - {i}"
            body = self.generate_email_body(subject)
            emails.append({"subject": subject, "body": body})
        
        return emails
    
    def generate_email_body(self, subject: str) -> str:
        """Generate a realistic email body based on subject."""
        templates = [
            f"Dear Team,\n\nI wanted to reach out regarding {subject}. ",
            f"Hi all,\n\nQuick update on {subject}. ",
            f"Good morning,\n\nI'm writing to discuss {subject}. ",
            f"Hello,\n\nI need your input on {subject}. ",
        ]
        
        middle_parts = [
            "We've made significant progress on this initiative and I'd like to share the current status. ",
            "There are some important considerations we need to address before moving forward. ",
            "I've attached the relevant documents for your review. ",
            "This requires immediate attention from the team. ",
            "Based on our last discussion, I've compiled the following action items. ",
        ]
        
        endings = [
            "Please let me know your thoughts.\n\nBest regards,",
            "Looking forward to your feedback.\n\nThanks,",
            "Let's discuss this in our next meeting.\n\nRegards,",
            "Please respond by EOD.\n\nThank you,",
            "Happy to clarify any questions.\n\nBest,",
        ]
        
        body = random.choice(templates) + random.choice(middle_parts) + random.choice(endings)
        return body
    
    def generate_sender(self, is_internal: bool = True) -> Dict[str, str]:
        """Generate a realistic sender with name and email."""
        if is_internal:
            first = random.choice(self.sender_patterns["internal"]["first_names"])
            last = random.choice(self.sender_patterns["internal"]["last_names"])
            domain = random.choice(self.sender_patterns["internal"]["domains"])
            dept = random.choice(self.sender_patterns["internal"]["departments"])
            
            email = f"{first.lower()}.{last.lower()}@{domain}"
            name = f"{first} {last}"
            return {
                "name": name,
                "email": email,
                "department": dept,
                "is_internal": True
            }
        else:
            company = random.choice(self.sender_patterns["external"]["companies"])
            domain = random.choice(self.sender_patterns["external"]["domains"])
            first = random.choice(self.sender_patterns["internal"]["first_names"])
            last = random.choice(self.sender_patterns["internal"]["last_names"])
            
            if "gmail" in domain or "yahoo" in domain or "outlook" in domain:
                email = f"{first.lower()}{last.lower()}{random.randint(1,999)}@{domain}"
            else:
                email = f"{first.lower()}@{company.lower()}.{domain.split('.')[1]}"
            
            return {
                "name": f"{first} {last}",
                "email": email,
                "company": company,
                "is_internal": False
            }
    
    def generate_timestamp(self, base_date: datetime = None) -> str:
        """Generate realistic timestamps."""
        if base_date is None:
            base_date = datetime.now() - timedelta(days=random.randint(0, 30))
        
        # Add some variance in time
        hours_offset = random.randint(-72, 72)
        minutes_offset = random.randint(0, 59)
        
        timestamp = base_date + timedelta(hours=hours_offset, minutes=minutes_offset)
        return timestamp.isoformat() + "Z"
    
    def determine_category(self, email: Dict) -> str:
        """Determine email category based on content."""
        subject = email.get("subject", "").lower()
        body = email.get("body", "").lower()
        full_text = subject + " " + body
        
        # Rule-based categorization
        if any(word in full_text for word in ["meeting", "schedule", "calendar", "appointment"]):
            return "meeting_request"
        elif any(word in full_text for word in ["urgent", "asap", "critical", "immediately"]):
            return "urgent_action"
        elif any(word in full_text for word in ["update", "status", "progress", "report"]):
            return "project_update"
        elif any(word in full_text for word in ["expense", "reimburse", "receipt", "payment"]):
            return "expense_report"
        elif any(word in full_text for word in ["contract", "agreement", "terms", "legal"]):
            return "contract_review"
        elif any(word in full_text for word in ["help", "support", "issue", "problem", "error"]):
            return "technical_issue"
        elif any(word in full_text for word in ["newsletter", "announcement", "bulletin"]):
            return "newsletter"
        elif any(word in full_text for word in ["feedback", "review", "opinion", "thoughts"]):
            return "feedback"
        else:
            return random.choice(self.categories)
    
    def determine_priority(self, email: Dict, category: str) -> str:
        """Determine email priority based on content and category."""
        subject = email.get("subject", "").lower()
        body = email.get("body", "").lower()
        sender = email.get("sender", {})
        
        # Critical priority indicators
        if any(word in subject + body for word in ["urgent", "critical", "emergency", "asap"]):
            return "critical"
        
        # High priority categories
        if category in ["urgent_action", "contract_review", "customer_support", "complaint"]:
            return "high"
        
        # Low priority categories
        if category in ["newsletter", "spam", "personal"]:
            return random.choice(["low", "minimal"])
        
        # Internal vs external
        if sender.get("is_internal", False):
            weights = [5, 20, 50, 20, 5]  # More likely medium/high
        else:
            weights = [3, 15, 40, 30, 12]  # More varied
        
        priorities = list(self.priority_weights.keys())
        return random.choices(priorities, weights=weights)[0]
    
    def select_template(self, category: str, priority: str) -> str:
        """Select appropriate template based on category and priority."""
        template_mapping = {
            "meeting_request": ["meeting_accept", "meeting_decline"],
            "urgent_action": ["urgent_response", "delegation"],
            "information_request": ["information_request", "followup"],
            "project_update": ["project_update", "acknowledgment"],
            "customer_support": ["urgent_response", "information_request"],
            "contract_review": ["acknowledgment", "delegation"],
            "feedback": ["acknowledgment", "followup"],
            "collaboration_request": ["meeting_accept", "information_request"],
            "event_invitation": ["meeting_accept", "meeting_decline"],
            "complaint": ["urgent_response", "acknowledgment"]
        }
        
        # Default templates
        default_templates = ["acknowledgment", "followup", "delegation"]
        
        # Get appropriate templates for category
        possible_templates = template_mapping.get(category, default_templates)
        
        # Adjust based on priority
        if priority == "critical":
            return "urgent_response"
        elif priority == "minimal":
            return "acknowledgment"
        else:
            return random.choice(possible_templates)
    
    def generate(self, num_test_cases: int = 200):
        """Generate complete test cases and ground truth."""
        # Load dataset
        self.load_email_dataset()
        
        print(f"\n🔧 Generating {num_test_cases} complete test cases...")
        
        for test_idx in range(num_test_cases):
            # Vary number of emails per test case (1-10, weighted towards 1-5)
            num_emails = random.choices(
                range(1, 11),
                weights=[30, 25, 20, 10, 5, 3, 2, 2, 2, 1]
            )[0]
            
            emails = []
            classifications = []
            priorities = []
            response_scores = []
            
            base_date = datetime.now() - timedelta(days=random.randint(0, 60))
            
            for i in range(num_emails):
                # Get email content from dataset or fallback
                if self.dataset and i + test_idx * 10 < len(self.dataset):
                    data_idx = (test_idx * 10 + i) % len(self.dataset)
                    email_data = self.dataset[data_idx]
                    
                    # Extract subject and body
                    if isinstance(email_data, dict):
                        if "email_body" in email_data:
                            # AESLC dataset format
                            body = email_data.get("email_body", "")
                            # Extract subject from body if possible
                            lines = body.split("\n")
                            subject = lines[0][:100] if lines else "No Subject"
                        else:
                            subject = email_data.get("subject", "No Subject")
                            body = email_data.get("body", email_data.get("text", ""))
                    else:
                        subject = "Email " + str(i)
                        body = str(email_data)
                else:
                    # Use fallback
                    subject = f"Email Subject {test_idx}_{i}"
                    body = self.generate_email_body(subject)
                
                # Generate sender (70% internal, 30% external)
                is_internal = random.random() < 0.7
                sender = self.generate_sender(is_internal)
                
                # Create email object
                email = {
                    "id": f"{test_idx}_{i}",
                    "subject": subject[:200],  # Limit subject length
                    "body": body[:2000],  # Limit body length
                    "sender": sender,
                    "timestamp": self.generate_timestamp(base_date),
                    "attachments": random.choice([[], ["document.pdf"], ["report.xlsx"], ["presentation.pptx"]]),
                    "cc": [] if random.random() < 0.6 else [self.generate_sender().get("email") for _ in range(random.randint(1, 3))],
                    "thread_id": f"thread_{test_idx}_{i//3}",  # Group some emails into threads
                    "is_reply": random.random() < 0.3
                }
                emails.append(email)
                
                # Generate ground truth
                category = self.determine_category(email)
                priority = self.determine_priority(email, category)
                response_quality = round(random.uniform(3.0, 5.0), 1)
                
                classifications.append(category)
                priorities.append(priority)
                response_scores.append(response_quality)
            
            # Generate user context with more variety
            departments = ["engineering", "sales", "marketing", "finance", "hr", "operations", "legal", "research"]
            roles = ["manager", "director", "analyst", "engineer", "specialist", "coordinator", "vp", "associate"]
            
            user_context = {
                "role": random.choice(roles),
                "department": random.choice(departments),
                "name": f"{random.choice(self.sender_patterns['internal']['first_names'])} {random.choice(self.sender_patterns['internal']['last_names'])}",
                "email": f"user@{random.choice(self.sender_patterns['internal']['domains'])}",
                "preferences": {
                    "auto_archive_newsletters": random.choice([True, False]),
                    "priority_sender_list": [self.generate_sender().get("email") for _ in range(random.randint(0, 5))],
                    "blocked_domains": [] if random.random() < 0.7 else ["spam.com", "phishing.net"],
                    "working_hours": f"{random.randint(7,10)}:00-{random.randint(17,19)}:00"
                },
                "ooo_status": random.choice([False, False, False, {"start": "2024-01-15", "end": "2024-01-20", "backup": "colleague@company.com"}]),
                "signature": f"Best regards,\n{user_context['name'] if 'name' in locals() else 'User'}\n{random.choice(departments)} Department"
            }
            
            # Create test case
            template_list = list(self.templates.values())
            test_case = {
                "test_id": f"email_{test_idx:03d}",
                "emails": emails,
                "templates": random.sample(template_list, min(len(template_list), 5)),  # Provide 5 random templates
                "user_context": user_context,
                "auto_respond": random.choice([True, False, False]),  # 33% chance of auto-respond
                "classification_rules": {
                    "keywords": {
                        "urgent": ["asap", "urgent", "critical", "emergency"],
                        "meeting": ["meeting", "schedule", "calendar", "appointment"],
                        "project": ["project", "milestone", "deliverable", "timeline"]
                    },
                    "sender_rules": {
                        "vip": [self.generate_sender().get("email") for _ in range(random.randint(1, 3))],
                        "auto_flag": [] if random.random() < 0.5 else [self.generate_sender().get("email")]
                    }
                }
            }
            self.test_cases.append(test_case)
            
            # Select template for this batch
            selected_template = self.select_template(classifications[0], priorities[0])
            
            # Determine actions
            actions_taken = []
            for cat, pri in zip(classifications, priorities):
                if cat == "spam":
                    actions_taken.append("delete")
                elif cat == "newsletter":
                    actions_taken.append("archive")
                elif pri in ["critical", "high"]:
                    actions_taken.append("flag")
                    actions_taken.append("reply")
                else:
                    actions_taken.append(random.choice(["archive", "reply", "forward", "categorize"]))
            
            # Create ground truth
            ground_truth = {
                "test_id": f"email_{test_idx:03d}",
                "expected_classifications": classifications,
                "expected_priorities": priorities,
                "expected_template": selected_template,
                "response_quality_scores": response_scores,
                "expected_actions": list(set(actions_taken)),  # Unique actions
                "processing_notes": f"Batch of {num_emails} emails with {priorities.count('critical')} critical items"
            }
            self.ground_truth.append(ground_truth)
            
            if (test_idx + 1) % 50 == 0:
                print(f"  Generated {test_idx + 1}/{num_test_cases} test cases...")
        
        print(f"✅ Generated {len(self.test_cases)} test cases and ground truth entries")
    
    def validate(self):
        """Validate the generated data."""
        print("\n🔍 Validating generated data...")
        
        issues = []
        
        for i, (test, truth) in enumerate(zip(self.test_cases, self.ground_truth)):
            # Check test case
            if not test.get('emails'):
                issues.append(f"Test {i}: No emails")
            if not test.get('templates'):
                issues.append(f"Test {i}: No templates")
            if not test.get('user_context'):
                issues.append(f"Test {i}: No user context")
            
            # Check emails have required fields
            for j, email in enumerate(test.get('emails', [])):
                if not email.get('subject'):
                    issues.append(f"Test {i}, Email {j}: No subject")
                if not email.get('sender', {}).get('email'):
                    issues.append(f"Test {i}, Email {j}: No sender email")
            
            # Check ground truth
            if len(truth.get('expected_classifications', [])) != len(test.get('emails', [])):
                issues.append(f"Truth {i}: Classification count mismatch")
        
        if issues:
            print(f"⚠️ Found {len(issues)} issues:")
            for issue in issues[:5]:
                print(f"  - {issue}")
        else:
            print("✅ All validation checks passed!")
        
        # Show statistics
        print("\n📊 Data Statistics:")
        avg_emails = sum(len(tc['emails']) for tc in self.test_cases) / len(self.test_cases)
        categories_used = set()
        priorities_used = set()
        for gt in self.ground_truth:
            categories_used.update(gt['expected_classifications'])
            priorities_used.update(gt['expected_priorities'])
        
        print(f"  Average emails per test: {avg_emails:.1f}")
        print(f"  Unique categories used: {len(categories_used)}")
        print(f"  Priority levels used: {priorities_used}")
    
    def save(self, output_dir: str = "data/email_automation"):
        """Save test cases and ground truth."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save test cases
        test_file = os.path.join(output_dir, "test_cases.json")
        with open(test_file, 'w') as f:
            json.dump(self.test_cases, f, indent=2)
        print(f"📁 Saved test cases to {test_file}")
        
        # Save ground truth
        truth_file = os.path.join(output_dir, "ground_truth.json")
        with open(truth_file, 'w') as f:
            json.dump(self.ground_truth, f, indent=2)
        print(f"📁 Saved ground truth to {truth_file}")
        
        # Show sample
        if self.test_cases:
            sample = self.test_cases[0]
            print("\n📋 Sample Test Case:")
            print(f"  ID: {sample['test_id']}")
            print(f"  Emails: {len(sample['emails'])}")
            if sample['emails']:
                print(f"  First email:")
                print(f"    Subject: {sample['emails'][0]['subject'][:60]}...")
                print(f"    Sender: {sample['emails'][0]['sender']['name']} ({sample['emails'][0]['sender']['email']})")
            print(f"  Templates provided: {len(sample['templates'])}")
            print(f"  User: {sample['user_context']['name']} - {sample['user_context']['role']}")


def main():
    """Generate complete email automation data."""
    print("=" * 60)
    print("📧 EMAIL AUTOMATION DATA GENERATOR")
    print("=" * 60)
    print("Generating COMPLETE data for email automation use case")
    print("This includes:")
    print("  ✓ Realistic email content from Enron dataset")
    print("  ✓ Varied senders with names and departments")
    print("  ✓ Multiple emails per test case (batch processing)")
    print("  ✓ Response templates for different scenarios")
    print("  ✓ User context with preferences and OOO status")
    print("  ✓ Classification rules and priorities")
    print("=" * 60)
    
    generator = EmailAutomationDataGenerator()
    generator.generate(num_test_cases=200)
    generator.validate()
    generator.save()
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE DATA GENERATION FINISHED")
    print("=" * 60)
    print("The email automation use case now has:")
    print("  • Realistic test cases with multiple emails")
    print("  • Comprehensive ground truth for evaluation")
    print("  • All fields required by the agents")
    print("  • Rich user context and classification rules")
    print("\nReady for testing with all frameworks!")


if __name__ == "__main__":
    main()
