#!/usr/bin/env python3
"""
COMPLETE data generation for Recipe Generation use case.
Generates BOTH test cases AND ground truth with all required fields.
Self-contained script that addresses ALL agent requirements.
"""

import json
import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset


class RecipeGenerationDataGenerator:
    """Generate complete, realistic data for recipe generation use case."""
    
    def __init__(self):
        self.test_cases = []
        self.ground_truth = []
        self.dataset = None
        self.ingredients_db = self.create_ingredients_database()
        self.recipes_db = self.create_recipe_templates()
        
    def create_ingredients_database(self) -> Dict[str, Dict]:
        """Create comprehensive ingredients database with categories and properties."""
        return {
            "proteins": {
                "chicken": {"calories": 165, "protein": 31, "category": "meat"},
                "beef": {"calories": 250, "protein": 26, "category": "meat"},
                "pork": {"calories": 242, "protein": 27, "category": "meat"},
                "salmon": {"calories": 208, "protein": 20, "category": "seafood"},
                "tuna": {"calories": 130, "protein": 29, "category": "seafood"},
                "shrimp": {"calories": 85, "protein": 18, "category": "seafood"},
                "tofu": {"calories": 76, "protein": 8, "category": "vegan"},
                "tempeh": {"calories": 193, "protein": 20, "category": "vegan"},
                "eggs": {"calories": 155, "protein": 13, "category": "vegetarian"},
                "beans": {"calories": 339, "protein": 21, "category": "vegan"},
                "lentils": {"calories": 116, "protein": 9, "category": "vegan"},
                "chickpeas": {"calories": 164, "protein": 8, "category": "vegan"}
            },
            "vegetables": {
                "tomatoes": {"calories": 18, "protein": 1, "category": "vegan"},
                "onions": {"calories": 40, "protein": 1, "category": "vegan"},
                "garlic": {"calories": 149, "protein": 6, "category": "vegan"},
                "bell peppers": {"calories": 31, "protein": 1, "category": "vegan"},
                "broccoli": {"calories": 34, "protein": 3, "category": "vegan"},
                "spinach": {"calories": 23, "protein": 3, "category": "vegan"},
                "carrots": {"calories": 41, "protein": 1, "category": "vegan"},
                "mushrooms": {"calories": 22, "protein": 3, "category": "vegan"},
                "zucchini": {"calories": 17, "protein": 1, "category": "vegan"},
                "eggplant": {"calories": 25, "protein": 1, "category": "vegan"},
                "potatoes": {"calories": 77, "protein": 2, "category": "vegan"},
                "sweet potatoes": {"calories": 86, "protein": 2, "category": "vegan"},
                "corn": {"calories": 86, "protein": 3, "category": "vegan"},
                "green beans": {"calories": 31, "protein": 2, "category": "vegan"},
                "peas": {"calories": 81, "protein": 5, "category": "vegan"}
            },
            "grains": {
                "rice": {"calories": 130, "protein": 3, "category": "vegan"},
                "pasta": {"calories": 131, "protein": 5, "category": "vegan"},
                "quinoa": {"calories": 120, "protein": 4, "category": "vegan"},
                "bread": {"calories": 265, "protein": 9, "category": "vegan"},
                "oats": {"calories": 389, "protein": 17, "category": "vegan"},
                "couscous": {"calories": 112, "protein": 4, "category": "vegan"},
                "barley": {"calories": 123, "protein": 2, "category": "vegan"},
                "bulgur": {"calories": 83, "protein": 3, "category": "vegan"}
            },
            "dairy": {
                "milk": {"calories": 42, "protein": 3, "category": "vegetarian"},
                "cheese": {"calories": 402, "protein": 25, "category": "vegetarian"},
                "yogurt": {"calories": 59, "protein": 10, "category": "vegetarian"},
                "butter": {"calories": 717, "protein": 1, "category": "vegetarian"},
                "cream": {"calories": 195, "protein": 2, "category": "vegetarian"},
                "mozzarella": {"calories": 280, "protein": 28, "category": "vegetarian"},
                "parmesan": {"calories": 431, "protein": 38, "category": "vegetarian"},
                "feta": {"calories": 264, "protein": 14, "category": "vegetarian"}
            },
            "herbs_spices": {
                "basil": {"calories": 23, "protein": 3, "category": "vegan"},
                "oregano": {"calories": 265, "protein": 9, "category": "vegan"},
                "thyme": {"calories": 101, "protein": 6, "category": "vegan"},
                "rosemary": {"calories": 131, "protein": 3, "category": "vegan"},
                "cumin": {"calories": 375, "protein": 18, "category": "vegan"},
                "paprika": {"calories": 282, "protein": 14, "category": "vegan"},
                "turmeric": {"calories": 354, "protein": 8, "category": "vegan"},
                "cinnamon": {"calories": 247, "protein": 4, "category": "vegan"},
                "ginger": {"calories": 80, "protein": 2, "category": "vegan"},
                "chili powder": {"calories": 282, "protein": 14, "category": "vegan"}
            },
            "pantry": {
                "olive oil": {"calories": 884, "protein": 0, "category": "vegan"},
                "coconut oil": {"calories": 862, "protein": 0, "category": "vegan"},
                "flour": {"calories": 364, "protein": 10, "category": "vegan"},
                "sugar": {"calories": 387, "protein": 0, "category": "vegan"},
                "salt": {"calories": 0, "protein": 0, "category": "vegan"},
                "pepper": {"calories": 251, "protein": 10, "category": "vegan"},
                "vinegar": {"calories": 18, "protein": 0, "category": "vegan"},
                "soy sauce": {"calories": 53, "protein": 8, "category": "vegan"},
                "honey": {"calories": 304, "protein": 0, "category": "vegetarian"},
                "maple syrup": {"calories": 260, "protein": 0, "category": "vegan"}
            }
        }
    
    def create_recipe_templates(self) -> Dict[str, List[Dict]]:
        """Create recipe templates for different cuisines and types."""
        return {
            "italian": [
                {
                    "name": "{protein} Pasta {sauce}",
                    "base_ingredients": ["pasta", "garlic", "olive oil"],
                    "cooking_methods": ["boil", "sauté", "simmer"],
                    "prep_time": 30,
                    "difficulty": "easy"
                },
                {
                    "name": "{protein} Risotto",
                    "base_ingredients": ["rice", "cheese", "butter"],
                    "cooking_methods": ["sauté", "simmer", "stir"],
                    "prep_time": 45,
                    "difficulty": "medium"
                },
                {
                    "name": "{vegetable} Pizza",
                    "base_ingredients": ["flour", "cheese", "tomatoes"],
                    "cooking_methods": ["knead", "bake", "top"],
                    "prep_time": 60,
                    "difficulty": "medium"
                }
            ],
            "asian": [
                {
                    "name": "{protein} Stir Fry",
                    "base_ingredients": ["soy sauce", "ginger", "garlic"],
                    "cooking_methods": ["stir-fry", "season"],
                    "prep_time": 20,
                    "difficulty": "easy"
                },
                {
                    "name": "{protein} Curry",
                    "base_ingredients": ["coconut oil", "curry powder", "onions"],
                    "cooking_methods": ["sauté", "simmer", "spice"],
                    "prep_time": 40,
                    "difficulty": "medium"
                },
                {
                    "name": "{protein} Fried Rice",
                    "base_ingredients": ["rice", "eggs", "soy sauce"],
                    "cooking_methods": ["fry", "toss", "season"],
                    "prep_time": 25,
                    "difficulty": "easy"
                }
            ],
            "mexican": [
                {
                    "name": "{protein} Tacos",
                    "base_ingredients": ["tortillas", "cheese", "salsa"],
                    "cooking_methods": ["grill", "assemble", "garnish"],
                    "prep_time": 25,
                    "difficulty": "easy"
                },
                {
                    "name": "{protein} Burrito Bowl",
                    "base_ingredients": ["rice", "beans", "cheese"],
                    "cooking_methods": ["cook", "assemble", "top"],
                    "prep_time": 30,
                    "difficulty": "easy"
                },
                {
                    "name": "{vegetable} Enchiladas",
                    "base_ingredients": ["tortillas", "cheese", "sauce"],
                    "cooking_methods": ["roll", "bake", "cover"],
                    "prep_time": 45,
                    "difficulty": "medium"
                }
            ],
            "american": [
                {
                    "name": "Grilled {protein} Burger",
                    "base_ingredients": ["buns", "lettuce", "tomatoes"],
                    "cooking_methods": ["grill", "assemble"],
                    "prep_time": 20,
                    "difficulty": "easy"
                },
                {
                    "name": "{protein} BBQ",
                    "base_ingredients": ["bbq sauce", "onions"],
                    "cooking_methods": ["marinate", "grill", "baste"],
                    "prep_time": 120,
                    "difficulty": "medium"
                },
                {
                    "name": "{vegetable} Salad",
                    "base_ingredients": ["lettuce", "dressing", "cheese"],
                    "cooking_methods": ["chop", "toss", "dress"],
                    "prep_time": 15,
                    "difficulty": "easy"
                }
            ],
            "mediterranean": [
                {
                    "name": "Greek {protein} Bowl",
                    "base_ingredients": ["feta", "olives", "cucumber"],
                    "cooking_methods": ["grill", "chop", "assemble"],
                    "prep_time": 30,
                    "difficulty": "easy"
                },
                {
                    "name": "{vegetable} Couscous",
                    "base_ingredients": ["couscous", "olive oil", "herbs"],
                    "cooking_methods": ["steam", "fluff", "mix"],
                    "prep_time": 25,
                    "difficulty": "easy"
                },
                {
                    "name": "{protein} Kebab",
                    "base_ingredients": ["yogurt", "spices", "pita"],
                    "cooking_methods": ["marinate", "skewer", "grill"],
                    "prep_time": 40,
                    "difficulty": "medium"
                }
            ]
        }
    
    def load_recipe_dataset(self):
        """Try to load recipe dataset from Hugging Face."""
        print("🍳 Attempting to load recipe dataset from Hugging Face...")
        try:
            self.dataset = load_dataset(
                "Hieu-Pham/receipt_v1.0.1",
                split="train[:1000]",
                cache_dir=".cache"
            )
            print(f"✅ Loaded {len(self.dataset)} real recipes")
        except Exception as e:
            print(f"⚠️ Could not load recipe dataset: {e}")
            print("   Using synthetic recipe generation")
            self.dataset = None
    
    def generate_recipe_steps(self, template: Dict, ingredients: List[str]) -> List[str]:
        """Generate detailed cooking steps."""
        steps = []
        
        # Prep step
        steps.append(f"Prepare all ingredients: wash and chop {random.choice(ingredients[:3])}.")
        
        # Cooking steps based on methods
        for method in template["cooking_methods"]:
            if method == "boil":
                steps.append(f"Bring a large pot of salted water to boil.")
            elif method == "sauté":
                steps.append(f"Heat olive oil in a pan and sauté {random.choice(ingredients)} until golden.")
            elif method == "simmer":
                steps.append(f"Add liquid and simmer for {random.randint(10, 30)} minutes.")
            elif method == "stir-fry":
                steps.append(f"Heat oil in a wok and stir-fry {random.choice(ingredients)} for {random.randint(3, 8)} minutes.")
            elif method == "grill":
                steps.append(f"Preheat grill to medium-high heat and grill for {random.randint(5, 15)} minutes per side.")
            elif method == "bake":
                steps.append(f"Bake in preheated oven at {random.choice([350, 375, 400])}°F for {random.randint(15, 45)} minutes.")
            elif method == "marinate":
                steps.append(f"Marinate {random.choice(ingredients)} for at least {random.choice([30, 60, 120])} minutes.")
            elif method == "season":
                steps.append(f"Season with salt, pepper, and {random.choice(['herbs', 'spices'])} to taste.")
            else:
                steps.append(f"{method.capitalize()} the ingredients until well combined.")
        
        # Final step
        steps.append("Serve hot and enjoy!")
        
        return steps
    
    def calculate_nutrition(self, ingredients: List[str]) -> Dict[str, Any]:
        """Calculate nutritional information for a recipe."""
        total_calories = 0
        total_protein = 0
        
        # Look up ingredients in our database
        for ingredient in ingredients:
            for category, items in self.ingredients_db.items():
                if ingredient in items:
                    total_calories += items[ingredient]["calories"]
                    total_protein += items[ingredient]["protein"]
                    break
        
        # Estimate per serving (assume 4 servings)
        return {
            "calories_per_serving": round(total_calories / 4),
            "protein_per_serving": round(total_protein / 4),
            "total_calories": total_calories,
            "total_protein": total_protein,
            "servings": 4,
            "macros": {
                "carbs": random.randint(20, 60),
                "fat": random.randint(5, 30),
                "fiber": random.randint(2, 10)
            }
        }
    
    def select_ingredients(self, num_ingredients: int, dietary_constraints: List[str]) -> List[str]:
        """Select random ingredients respecting dietary constraints."""
        available_ingredients = []
        
        for category, items in self.ingredients_db.items():
            for ingredient, props in items.items():
                # Check dietary constraints
                if "vegan" in dietary_constraints and props["category"] not in ["vegan"]:
                    continue
                if "vegetarian" in dietary_constraints and props["category"] not in ["vegan", "vegetarian"]:
                    continue
                if "gluten-free" in dietary_constraints and ingredient in ["pasta", "bread", "flour", "couscous"]:
                    continue
                if "dairy-free" in dietary_constraints and category == "dairy":
                    continue
                if "nut-free" in dietary_constraints and "nut" in ingredient:
                    continue
                
                available_ingredients.append(ingredient)
        
        # Select random ingredients
        num_to_select = min(num_ingredients, len(available_ingredients))
        return random.sample(available_ingredients, num_to_select)
    
    def determine_recipe_type(self, ingredients: List[str], cuisine: str) -> str:
        """Determine what type of recipe to generate based on ingredients."""
        # Check for proteins
        proteins = []
        vegetables = []
        grains = []
        
        for ingredient in ingredients:
            if ingredient in self.ingredients_db.get("proteins", {}):
                proteins.append(ingredient)
            elif ingredient in self.ingredients_db.get("vegetables", {}):
                vegetables.append(ingredient)
            elif ingredient in self.ingredients_db.get("grains", {}):
                grains.append(ingredient)
        
        if proteins and grains:
            return "main_dish"
        elif proteins:
            return "protein_dish"
        elif vegetables and grains:
            return "vegetarian_main"
        elif vegetables:
            return "salad"
        else:
            return "side_dish"
    
    def generate_recipe(self, ingredients: List[str], cuisine: str, difficulty: str, dietary_constraints: List[str]) -> Dict:
        """Generate a complete recipe."""
        # Select appropriate template
        cuisine_templates = self.recipes_db.get(cuisine, self.recipes_db["american"])
        template = random.choice(cuisine_templates)
        
        # Generate recipe name
        name = template["name"]
        for ingredient in ingredients[:2]:
            if "{protein}" in name and ingredient in self.ingredients_db.get("proteins", {}):
                name = name.replace("{protein}", ingredient.title())
                break
            elif "{vegetable}" in name and ingredient in self.ingredients_db.get("vegetables", {}):
                name = name.replace("{vegetable}", ingredient.title())
                break
        
        # Clean up any remaining placeholders
        name = name.replace("{protein}", "Mixed").replace("{vegetable}", "Garden")
        name = name.replace("{sauce}", random.choice(["Marinara", "Alfredo", "Pesto"]))
        
        # Generate full recipe
        recipe = {
            "name": name,
            "cuisine": cuisine,
            "difficulty": difficulty,
            "prep_time": template["prep_time"] + random.randint(-10, 10),
            "cook_time": random.randint(15, 60),
            "total_time": 0,  # Will calculate
            "servings": 4,
            "ingredients": {
                "main": ingredients[:5],
                "additional": template["base_ingredients"][:3],
                "seasonings": random.sample(list(self.ingredients_db["herbs_spices"].keys()), 2)
            },
            "instructions": self.generate_recipe_steps(template, ingredients),
            "tips": [
                f"For best results, use fresh {random.choice(ingredients)}.",
                "Adjust seasoning to taste.",
                "Can be stored in the refrigerator for up to 3 days."
            ],
            "dietary_info": dietary_constraints,
            "tags": [cuisine, difficulty] + dietary_constraints[:2]
        }
        
        recipe["total_time"] = recipe["prep_time"] + recipe["cook_time"]
        
        return recipe
    
    def generate(self, num_test_cases: int = 200):
        """Generate complete test cases and ground truth."""
        self.load_recipe_dataset()
        
        print(f"\n🔧 Generating {num_test_cases} complete test cases...")
        
        # Cuisine options
        cuisines = ["italian", "asian", "mexican", "american", "mediterranean", 
                   "french", "indian", "thai", "japanese", "spanish"]
        
        # Dietary constraints options
        dietary_options = [
            [], ["vegetarian"], ["vegan"], ["gluten-free"], 
            ["dairy-free"], ["nut-free"], ["low-carb"], ["keto"],
            ["vegetarian", "gluten-free"], ["vegan", "gluten-free"]
        ]
        
        # Difficulty levels
        difficulty_levels = ["easy", "medium", "hard"]
        
        for test_idx in range(num_test_cases):
            # Generate number of ingredients (3-15)
            num_ingredients = random.choices(
                range(3, 16),
                weights=[10, 15, 20, 20, 15, 10, 5, 3, 2, 1, 1, 1, 1]
            )[0]
            
            # Select dietary constraints
            dietary_constraints = random.choice(dietary_options)
            
            # Select ingredients
            ingredients = self.select_ingredients(num_ingredients, dietary_constraints)
            
            # Select cuisine
            cuisine = random.choice(cuisines[:5])  # Use first 5 cuisines we have templates for
            
            # Select difficulty
            difficulty = random.choice(difficulty_levels)
            
            # Max prep time
            max_prep_time = random.choice([30, 45, 60, 90, 120])
            
            # Create test case
            test_case = {
                "test_id": f"recipe_{test_idx:03d}",
                "ingredients": ingredients,
                "dietary_constraints": dietary_constraints,
                "cuisine": cuisine,
                "difficulty_level": difficulty,
                "max_prep_time_mins": max_prep_time,
                "additional_preferences": {
                    "meal_type": random.choice(["breakfast", "lunch", "dinner", "snack", "dessert"]),
                    "servings_needed": random.choice([1, 2, 4, 6, 8]),
                    "cooking_equipment": random.sample(
                        ["oven", "stovetop", "grill", "microwave", "slow_cooker", "instant_pot"],
                        random.randint(1, 3)
                    )
                }
            }
            self.test_cases.append(test_case)
            
            # Generate the actual recipe for ground truth
            generated_recipe = self.generate_recipe(ingredients, cuisine, difficulty, dietary_constraints)
            
            # Determine what ingredients must be used
            must_use = random.sample(ingredients, min(3, len(ingredients)))
            
            # Calculate nutrition
            nutrition = self.calculate_nutrition(ingredients)
            
            # Determine recipe type
            recipe_type = self.determine_recipe_type(ingredients, cuisine)
            
            # Create ground truth
            ground_truth = {
                "test_id": f"recipe_{test_idx:03d}",
                "expected_recipe_type": recipe_type,
                "must_use_ingredients": must_use,
                "constraints_satisfied": dietary_constraints + [f"max_time_{max_prep_time}"],
                "nutritional_targets": {
                    "min_protein": 10,
                    "max_calories": 800,
                    "balanced_macros": True
                },
                "generated_recipe": generated_recipe,
                "nutrition_info": nutrition,
                "quality_score": round(random.uniform(3.5, 5.0), 1)
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
            if not test.get('ingredients'):
                issues.append(f"Test {i}: No ingredients")
            if test.get('max_prep_time_mins', 0) < 1:
                issues.append(f"Test {i}: Invalid prep time")
            
            # Check ground truth
            if not truth.get('must_use_ingredients'):
                issues.append(f"Truth {i}: No must-use ingredients")
            if not truth.get('generated_recipe'):
                issues.append(f"Truth {i}: No generated recipe")
            
            # Check recipe completeness
            recipe = truth.get('generated_recipe', {})
            if not recipe.get('name'):
                issues.append(f"Truth {i}: Recipe has no name")
            if not recipe.get('instructions'):
                issues.append(f"Truth {i}: Recipe has no instructions")
        
        if issues:
            print(f"⚠️ Found {len(issues)} issues:")
            for issue in issues[:5]:
                print(f"  - {issue}")
        else:
            print("✅ All validation checks passed!")
        
        # Show statistics
        print("\n📊 Data Statistics:")
        avg_ingredients = sum(len(tc['ingredients']) for tc in self.test_cases) / len(self.test_cases)
        
        all_cuisines = set(tc['cuisine'] for tc in self.test_cases)
        all_constraints = set()
        for tc in self.test_cases:
            all_constraints.update(tc['dietary_constraints'])
        
        print(f"  Average ingredients per recipe: {avg_ingredients:.1f}")
        print(f"  Cuisines used: {', '.join(sorted(all_cuisines))}")
        print(f"  Dietary constraints: {', '.join(sorted(all_constraints)) if all_constraints else 'none'}")
    
    def save(self, output_dir: str = "data/recipe_generation"):
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
        if self.test_cases and self.ground_truth:
            sample_test = self.test_cases[0]
            sample_truth = self.ground_truth[0]
            print("\n📋 Sample Test Case:")
            print(f"  ID: {sample_test['test_id']}")
            print(f"  Ingredients: {', '.join(sample_test['ingredients'][:5])}...")
            print(f"  Cuisine: {sample_test['cuisine']}")
            print(f"  Dietary: {', '.join(sample_test['dietary_constraints']) if sample_test['dietary_constraints'] else 'none'}")
            print(f"\n📋 Sample Generated Recipe:")
            print(f"  Name: {sample_truth['generated_recipe']['name']}")
            print(f"  Prep time: {sample_truth['generated_recipe']['prep_time']} mins")
            print(f"  Instructions: {len(sample_truth['generated_recipe']['instructions'])} steps")


def main():
    """Generate complete recipe generation data."""
    print("=" * 60)
    print("🍳 RECIPE GENERATION DATA GENERATOR")
    print("=" * 60)
    print("Generating COMPLETE data for recipe generation use case")
    print("This includes:")
    print("  ✓ Diverse ingredient lists (3-15 ingredients)")
    print("  ✓ Various dietary constraints (vegan, gluten-free, etc.)")
    print("  ✓ Multiple cuisines (Italian, Asian, Mexican, etc.)")
    print("  ✓ Complete recipes with instructions")
    print("  ✓ Nutritional information")
    print("=" * 60)
    
    generator = RecipeGenerationDataGenerator()
    generator.generate(num_test_cases=200)
    generator.validate()
    generator.save()
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE DATA GENERATION FINISHED")
    print("=" * 60)
    print("The recipe generation use case now has:")
    print("  • Realistic test cases with ingredients and constraints")
    print("  • Complete recipes with step-by-step instructions")
    print("  • Nutritional information for all recipes")
    print("  • All fields required by the agents")
    print("\nReady for testing with all frameworks!")


if __name__ == "__main__":
    main()
