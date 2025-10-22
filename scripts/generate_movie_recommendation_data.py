#!/usr/bin/env python3
"""
COMPLETE data generation for Movie Recommendation use case.
Generates BOTH test cases AND ground truth with all required fields.
Self-contained script that addresses ALL agent requirements.
"""

import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from datasets import load_dataset


class MovieRecommendationDataGenerator:
    """Generate complete, realistic data for movie recommendation use case."""
    
    def __init__(self):
        self.test_cases = []
        self.ground_truth = []
        self.dataset = None
        self.movies_db = self.create_movie_database()
        
    def create_movie_database(self) -> Dict[int, Dict]:
        """Create a comprehensive movie database."""
        # Movie titles by genre
        movie_templates = {
            "action": [
                "The {adj} {noun}", "Mission: {noun}", "{noun} Protocol", 
                "Edge of {noun}", "{adj} Strike", "The {noun} Initiative",
                "Code {noun}", "Operation {noun}", "{noun} Rising",
                "Last {noun}", "{adj} Fury", "The {noun} Conspiracy"
            ],
            "comedy": [
                "The {adj} {noun}", "Meet the {noun}s", "{adj} Days",
                "My {adj} {noun}", "Life of {noun}", "The {noun} Show",
                "Crazy {noun}", "{noun} Academy", "Super {noun}",
                "The {adj} Life", "{noun} Vacation", "Wedding {noun}"
            ],
            "drama": [
                "The {noun}", "A {adj} Life", "{noun}'s Story",
                "The {adj} Truth", "Beyond {noun}", "The {noun} Within",
                "Broken {noun}", "The Last {noun}", "Moments of {noun}",
                "The {adj} Road", "Letters from {noun}", "The {noun} Effect"
            ],
            "horror": [
                "The {adj} {noun}", "Night of the {noun}", "{noun} House",
                "The {noun} Within", "Dark {noun}", "The {adj} Presence",
                "Haunted {noun}", "The {noun} Below", "{adj} Shadows",
                "Curse of the {noun}", "The {noun} Files", "Dead {noun}"
            ],
            "sci-fi": [
                "{noun} Station", "The {adj} Dimension", "Planet {noun}",
                "Space {noun}", "The {noun} Paradox", "{adj} Future",
                "Quantum {noun}", "The {noun} Experiment", "Beyond {noun}",
                "Time {noun}", "The {adj} Colony", "Star {noun}"
            ],
            "romance": [
                "Love and {noun}", "The {adj} Heart", "{noun} in Love",
                "Forever {adj}", "The {noun} of Love", "My {adj} Romance",
                "Hearts and {noun}", "The {adj} Kiss", "Love's {noun}",
                "The {noun} Letter", "{adj} Moments", "When {noun} Met {noun}"
            ],
            "thriller": [
                "The {adj} Game", "{noun} Protocol", "The {noun} Files",
                "Code {noun}", "The {adj} Witness", "{noun} Theory",
                "The {noun} Conspiracy", "Silent {noun}", "The {adj} Hunt",
                "Deadly {noun}", "The {noun} Agenda", "Final {noun}"
            ],
            "documentary": [
                "The {noun} Story", "Inside {noun}", "The Truth About {noun}",
                "{adj} World", "The {noun} Revolution", "Beyond {noun}",
                "The Making of {noun}", "{noun}: A Journey", "The {adj} Era",
                "Secrets of {noun}", "The {noun} Movement", "Understanding {noun}"
            ],
            "animation": [
                "The {adj} {noun}", "{noun}'s Adventure", "Super {noun}",
                "The {noun} Movie", "{adj} Tales", "Adventure of {noun}",
                "The {noun} Kingdom", "{noun} and Friends", "Magic {noun}",
                "The {adj} Journey", "{noun} Island", "The {noun} Quest"
            ],
            "fantasy": [
                "The {adj} Kingdom", "{noun} of Magic", "The {noun} Chronicles",
                "Realm of {noun}", "The {adj} Quest", "{noun} and Dragons",
                "The {noun} Prophecy", "Magic {noun}", "The {adj} Realm",
                "Legend of {noun}", "The {noun} Saga", "{adj} Lands"
            ]
        }
        
        adjectives = [
            "Dark", "Silent", "Forgotten", "Lost", "Hidden", "Secret", "Ancient", "Modern",
            "Eternal", "Final", "Ultimate", "Mysterious", "Dangerous", "Beautiful", "Strange",
            "Wild", "Brave", "Bold", "Fierce", "Gentle", "Quiet", "Loud", "Fast", "Slow",
            "Big", "Small", "New", "Old", "First", "Last", "Golden", "Silver", "Red", "Blue"
        ]
        
        nouns = [
            "Knight", "Warrior", "Detective", "Spy", "Hero", "Villain", "King", "Queen",
            "Dragon", "Phoenix", "Storm", "Shadow", "Light", "Darkness", "Dream", "Reality",
            "Journey", "Quest", "Mission", "Adventure", "Mystery", "Secret", "Truth", "Lie",
            "Love", "Hate", "War", "Peace", "City", "World", "Universe", "Dimension",
            "Code", "Signal", "Message", "Letter", "Book", "Story", "Legend", "Myth"
        ]
        
        directors = [
            "Christopher Nolan", "Steven Spielberg", "Martin Scorsese", "Quentin Tarantino",
            "David Fincher", "Denis Villeneuve", "Ridley Scott", "James Cameron",
            "Coen Brothers", "Wes Anderson", "Paul Thomas Anderson", "Greta Gerwig",
            "Jordan Peele", "Ari Aster", "Robert Eggers", "Damien Chazelle",
            "Alfonso Cuarón", "Guillermo del Toro", "Bong Joon-ho", "Park Chan-wook",
            "Hayao Miyazaki", "Makoto Shinkai", "Edgar Wright", "Guy Ritchie",
            "Sam Mendes", "Danny Boyle", "Darren Aronofsky", "Terrence Malick"
        ]
        
        actors = [
            "Tom Hanks", "Morgan Freeman", "Meryl Streep", "Leonardo DiCaprio",
            "Brad Pitt", "Angelina Jolie", "Robert Downey Jr.", "Scarlett Johansson",
            "Chris Evans", "Chris Hemsworth", "Jennifer Lawrence", "Emma Stone",
            "Ryan Gosling", "Ryan Reynolds", "Tom Cruise", "Will Smith",
            "Denzel Washington", "Samuel L. Jackson", "Natalie Portman", "Cate Blanchett",
            "Christian Bale", "Amy Adams", "Jake Gyllenhaal", "Anne Hathaway",
            "Matt Damon", "Ben Affleck", "George Clooney", "Julia Roberts"
        ]
        
        # Generate movies
        movies = {}
        movie_id = 1
        
        for genre, templates in movie_templates.items():
            for _ in range(50):  # 50 movies per genre = 500 total
                template = random.choice(templates)
                adj = random.choice(adjectives)
                noun = random.choice(nouns)
                
                # Handle templates with two nouns
                if template.count("{noun}") > 1:
                    noun2 = random.choice([n for n in nouns if n != noun])
                    title = template.replace("{adj}", adj).replace("{noun}", noun, 1).replace("{noun}", noun2)
                else:
                    title = template.replace("{adj}", adj).replace("{noun}", noun)
                
                year = random.randint(1980, 2024)
                rating = round(random.uniform(3.0, 9.5), 1)
                runtime = random.randint(80, 180)
                
                # Generate cast
                num_actors = random.randint(3, 6)
                cast = random.sample(actors, num_actors)
                
                # Generate subgenres
                all_genres = list(movie_templates.keys())
                subgenres = [genre]
                if random.random() < 0.4:  # 40% chance of additional genre
                    other_genre = random.choice([g for g in all_genres if g != genre])
                    subgenres.append(other_genre)
                
                movies[movie_id] = {
                    "id": movie_id,
                    "title": title,
                    "year": year,
                    "genres": subgenres,
                    "rating": rating,
                    "runtime": runtime,
                    "director": random.choice(directors),
                    "cast": cast,
                    "description": f"A {genre} film about {noun.lower()} in a {adj.lower()} world.",
                    "popularity": random.uniform(0, 100)
                }
                movie_id += 1
        
        return movies
    
    def load_movie_dataset(self):
        """Try to load MovieLens or similar dataset."""
        print("🎬 Attempting to load movie dataset from Hugging Face...")
        try:
            self.dataset = load_dataset(
                "MongoDB/embedded_movies",
                split="train[:1000]",
                cache_dir=".cache"
            )
            print(f"✅ Loaded {len(self.dataset)} real movies")
            # Enhance our database with real data
            self.enhance_with_real_data()
        except Exception as e:
            print(f"⚠️ Could not load external dataset: {e}")
            print("   Using synthetic movie database")
    
    def enhance_with_real_data(self):
        """Enhance movie database with real data if available."""
        if not self.dataset:
            return
        
        movie_id = len(self.movies_db) + 1
        dataset_to_use = self.dataset[:1000] if hasattr(self.dataset, '__getitem__') else list(self.dataset)[:1000]
        
        for item in dataset_to_use:
            if isinstance(item, dict):
                title = item.get('title', '') or item.get('original_title', '')
                if title:
                    # Handle genres field that might be a list or string
                    genres_raw = item.get('genres', 'drama')
                    if isinstance(genres_raw, str):
                        genres = genres_raw.split(',') if genres_raw else ['drama']
                    elif isinstance(genres_raw, list):
                        genres = genres_raw
                    else:
                        genres = ['drama']
                    
                    self.movies_db[movie_id] = {
                        "id": movie_id,
                        "title": title[:100],
                        "year": item.get('year', random.randint(1990, 2024)),
                        "genres": [g.strip().lower() if isinstance(g, str) else str(g).lower() for g in genres],
                        "rating": item.get('vote_average', random.uniform(5, 8)),
                        "runtime": item.get('runtime', 120),
                        "director": "Various",
                        "cast": [],
                        "description": (item.get('overview', '') or '')[:200],
                        "popularity": item.get('popularity', 50)
                    }
                    movie_id += 1
    
    def generate_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Generate a realistic user profile based on viewing patterns."""
        profile_types = {
            "action_lover": {
                "preferred_genres": ["action", "thriller", "sci-fi"],
                "disliked_genres": ["romance", "documentary"],
                "rating_bias": 0.5,  # Rates action movies higher
                "watch_frequency": "high"
            },
            "drama_enthusiast": {
                "preferred_genres": ["drama", "romance", "documentary"],
                "disliked_genres": ["horror", "action"],
                "rating_bias": 0,
                "watch_frequency": "medium"
            },
            "family_viewer": {
                "preferred_genres": ["animation", "comedy", "fantasy"],
                "disliked_genres": ["horror", "thriller"],
                "rating_bias": 0.3,
                "watch_frequency": "high"
            },
            "horror_fan": {
                "preferred_genres": ["horror", "thriller", "sci-fi"],
                "disliked_genres": ["romance", "comedy"],
                "rating_bias": 0.2,
                "watch_frequency": "medium"
            },
            "cinephile": {
                "preferred_genres": ["drama", "documentary", "thriller"],
                "disliked_genres": [],
                "rating_bias": -0.3,  # More critical
                "watch_frequency": "very_high"
            },
            "casual_viewer": {
                "preferred_genres": ["comedy", "action", "animation"],
                "disliked_genres": ["documentary"],
                "rating_bias": 0.4,
                "watch_frequency": "low"
            },
            "genre_mixer": {
                "preferred_genres": random.sample(["action", "comedy", "drama", "sci-fi", "thriller", "romance"], 3),
                "disliked_genres": random.sample(["horror", "documentary", "romance"], 1),
                "rating_bias": random.uniform(-0.2, 0.2),
                "watch_frequency": random.choice(["low", "medium", "high"])
            }
        }
        
        # Select profile type with some randomization
        profile_type = random.choices(
            list(profile_types.keys()),
            weights=[20, 15, 15, 10, 10, 20, 10]
        )[0]
        
        return profile_types[profile_type]
    
    def generate_watch_history(self, user_profile: Dict, num_movies: int) -> List[Dict]:
        """Generate watch history based on user profile."""
        history = []
        all_movies = list(self.movies_db.values())
        
        # Filter movies by user preferences
        preferred_movies = [
            m for m in all_movies 
            if any(g in user_profile["preferred_genres"] for g in m["genres"])
        ]
        
        # Add some movies from disliked genres (people sometimes watch things they don't love)
        other_movies = [
            m for m in all_movies
            if not any(g in user_profile["preferred_genres"] for g in m["genres"])
        ]
        
        # 70% preferred, 30% other
        num_preferred = int(num_movies * 0.7)
        num_other = num_movies - num_preferred
        
        selected_movies = []
        if preferred_movies:
            selected_movies.extend(random.sample(preferred_movies, min(num_preferred, len(preferred_movies))))
        if other_movies and num_other > 0:
            selected_movies.extend(random.sample(other_movies, min(num_other, len(other_movies))))
        
        for movie in selected_movies:
            # Generate user rating based on profile
            base_rating = movie["rating"]
            
            # Adjust rating based on genre preference
            if any(g in user_profile["preferred_genres"] for g in movie["genres"]):
                user_rating = min(10, base_rating + user_profile["rating_bias"] + random.uniform(-0.5, 1.0))
            elif any(g in user_profile["disliked_genres"] for g in movie["genres"]):
                user_rating = max(1, base_rating - 2 + random.uniform(-1.0, 0.5))
            else:
                user_rating = base_rating + random.uniform(-1.0, 1.0)
            
            history.append({
                "id": movie["id"],
                "title": movie["title"],
                "rating": round(user_rating, 1),
                "genres": movie["genres"],
                "year": movie["year"],
                "watched_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            })
        
        return history
    
    def generate_constraints(self, user_profile: Dict) -> Dict[str, Any]:
        """Generate viewing constraints based on profile."""
        constraints = {}
        
        # Genre exclusions
        if user_profile["disliked_genres"]:
            constraints["exclude_genres"] = user_profile["disliked_genres"]
        
        # Year range preferences
        year_pref = random.choice(["recent", "classic", "any", "specific"])
        if year_pref == "recent":
            constraints["year_range"] = {"min": 2015, "max": 2024}
        elif year_pref == "classic":
            constraints["year_range"] = {"min": 1980, "max": 2000}
        elif year_pref == "specific":
            decade = random.choice([1980, 1990, 2000, 2010])
            constraints["year_range"] = {"min": decade, "max": decade + 9}
        
        # Rating threshold
        if random.random() < 0.4:
            constraints["min_rating"] = random.choice([6.0, 7.0, 7.5])
        
        # Runtime preferences
        if random.random() < 0.3:
            runtime_pref = random.choice(["short", "standard", "long"])
            if runtime_pref == "short":
                constraints["max_runtime"] = 100
            elif runtime_pref == "long":
                constraints["min_runtime"] = 150
        
        # Content rating (if family viewer)
        if "animation" in user_profile["preferred_genres"]:
            constraints["content_rating"] = ["G", "PG", "PG-13"]
        
        return constraints
    
    def select_held_out_movies(self, user_profile: Dict, watch_history: List[Dict], num_movies: int = 5) -> List[int]:
        """Select movies that should be recommended (for ground truth)."""
        watched_ids = {m["id"] for m in watch_history}
        all_movies = list(self.movies_db.values())
        
        # Find unwatched movies that match user preferences
        candidates = [
            m for m in all_movies
            if m["id"] not in watched_ids
            and any(g in user_profile["preferred_genres"] for g in m["genres"])
            and not any(g in user_profile["disliked_genres"] for g in m["genres"])
        ]
        
        if len(candidates) < num_movies:
            # Add more candidates if needed
            more_candidates = [
                m for m in all_movies
                if m["id"] not in watched_ids and m not in candidates
            ]
            candidates.extend(more_candidates)
        
        # Sort by rating and popularity
        candidates.sort(key=lambda x: (x["rating"], x["popularity"]), reverse=True)
        
        # Select top movies
        return [m["id"] for m in candidates[:num_movies]]
    
    def generate(self, num_test_cases: int = 200):
        """Generate complete test cases and ground truth."""
        # Try to load real data
        self.load_movie_dataset()
        
        print(f"\n🔧 Generating {num_test_cases} complete test cases...")
        print(f"   Using movie database with {len(self.movies_db)} movies")
        
        for test_idx in range(num_test_cases):
            user_id = 1000 + test_idx
            
            # Generate user profile
            user_profile = self.generate_user_profile(user_id)
            
            # Generate watch history (10-50 movies)
            num_watched = random.choices(
                [10, 15, 20, 30, 40, 50],
                weights=[20, 25, 25, 15, 10, 5]
            )[0]
            watch_history = self.generate_watch_history(user_profile, num_watched)
            
            # Generate constraints
            constraints = self.generate_constraints(user_profile)
            
            # Number of recommendations requested
            num_recommendations = random.choices(
                [5, 10, 15, 20],
                weights=[40, 35, 20, 5]
            )[0]
            
            # Create test case
            test_case = {
                "test_id": f"movie_{test_idx:03d}",
                "user_id": user_id,
                "watch_history": watch_history,
                "constraints": constraints,
                "num_recommendations": num_recommendations,
                "metadata": {
                    "account_age_days": random.randint(30, 3650),
                    "last_active": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                    "subscription_type": random.choice(["basic", "standard", "premium"]),
                    "devices": random.sample(["mobile", "tablet", "tv", "laptop"], random.randint(1, 3))
                }
            }
            self.test_cases.append(test_case)
            
            # Generate ground truth
            held_out_movies = self.select_held_out_movies(
                user_profile, 
                watch_history, 
                min(num_recommendations, 10)
            )
            
            # Calculate expected genres based on history
            genre_counts = {}
            for movie in watch_history:
                for genre in movie["genres"]:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            expected_genres = sorted(genre_counts.keys(), key=lambda x: genre_counts[x], reverse=True)[:5]
            
            ground_truth = {
                "test_id": f"movie_{test_idx:03d}",
                "held_out_movies": held_out_movies,
                "expected_genres": expected_genres,
                "user_preferences": user_profile,
                "quality_metrics": {
                    "diversity_expected": random.uniform(0.3, 0.8),  # Genre diversity
                    "novelty_expected": random.uniform(0.2, 0.7),    # New vs familiar
                    "rating_threshold": 6.0 if "min_rating" not in constraints else constraints["min_rating"]
                }
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
            if not test.get('watch_history'):
                issues.append(f"Test {i}: No watch history")
            if test.get('num_recommendations', 0) < 1:
                issues.append(f"Test {i}: Invalid num_recommendations")
            
            # Check ground truth
            if not truth.get('held_out_movies'):
                issues.append(f"Truth {i}: No held out movies")
            if not truth.get('expected_genres'):
                issues.append(f"Truth {i}: No expected genres")
        
        if issues:
            print(f"⚠️ Found {len(issues)} issues:")
            for issue in issues[:5]:
                print(f"  - {issue}")
        else:
            print("✅ All validation checks passed!")
        
        # Show statistics
        print("\n📊 Data Statistics:")
        avg_history = sum(len(tc['watch_history']) for tc in self.test_cases) / len(self.test_cases)
        all_genres = set()
        for tc in self.test_cases:
            for movie in tc['watch_history']:
                all_genres.update(movie['genres'])
        
        print(f"  Average watch history size: {avg_history:.1f} movies")
        print(f"  Unique genres in dataset: {len(all_genres)}")
        print(f"  Genres: {', '.join(sorted(all_genres))}")
        print(f"  Total movies in database: {len(self.movies_db)}")
    
    def save(self, output_dir: str = "data/movie_recommendation"):
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
            print(f"  User ID: {sample['user_id']}")
            print(f"  Watch history: {len(sample['watch_history'])} movies")
            if sample['watch_history']:
                print(f"  Sample movie: {sample['watch_history'][0]['title']} (rating: {sample['watch_history'][0]['rating']})")
            print(f"  Recommendations requested: {sample['num_recommendations']}")


def main():
    """Generate complete movie recommendation data."""
    print("=" * 60)
    print("🎬 MOVIE RECOMMENDATION DATA GENERATOR")
    print("=" * 60)
    print("Generating COMPLETE data for movie recommendation use case")
    print("This includes:")
    print("  ✓ Rich movie database with 500+ movies")
    print("  ✓ User watch history with ratings")
    print("  ✓ User profiles and preferences")
    print("  ✓ Viewing constraints (genres, years, ratings)")
    print("  ✓ Held-out movies for evaluation")
    print("=" * 60)
    
    generator = MovieRecommendationDataGenerator()
    generator.generate(num_test_cases=200)
    generator.validate()
    generator.save()
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE DATA GENERATION FINISHED")
    print("=" * 60)
    print("The movie recommendation use case now has:")
    print("  • Realistic test cases with watch history")
    print("  • Comprehensive ground truth for evaluation")
    print("  • Rich user profiles and constraints")
    print("  • All fields required by the agents")
    print("\nReady for testing with all frameworks!")


if __name__ == "__main__":
    main()
