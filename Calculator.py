import math
import re

class AdvancedCalculator:
    def __init__(self):
        self.history = []
        self.memory = 0
        self.current_mode = "standard"
        
    def display_menu(self):
        """Display the main menu with available calculator modes"""
        print("\n" + "="*50)
        print("           ADVANCED PYTHON CALCULATOR")
        print("="*50)
        print("Available Modes:")
        print("1. Standard Calculator (Basic operations)")
        print("2. Scientific Calculator (Trig, log, etc.)")
        print("3. Programming Calculator (Base conversions)")
        print("4. Financial Calculator (Interest, loans)")
        print("5. Statistics Calculator (Mean, median, etc.)")
        print("6. View Calculation History")
        print("7. Memory Functions")
        print("8. Exit")
        print("-"*50)
        
    def get_user_choice(self):
        """Get and validate user choice with error handling"""
        while True:
            try:
                choice = int(input("Select a mode (1-8): "))
                if 1 <= choice <= 8:
                    return choice
                else:
                    print("Please enter a number between 1 and 8.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def standard_calculator(self):
        """Standard calculator with basic operations"""
        print("\n--- STANDARD CALCULATOR ---")
        print("Operations: +, -, *, /, %, ^ (power)")
        print("Type 'back' to return to main menu")
        
        while True:
            try:
                expression = input("\nEnter expression (e.g., 5 + 3): ").strip()
                if expression.lower() == 'back':
                    break
                
                # Validate expression format
                if not re.match(r'^[-+]?[0-9]*\.?[0-9]+\s*[-+*/%^]\s*[-+]?[0-9]*\.?[0-9]+$', expression):
                    print("Invalid expression format. Use: number operator number")
                    continue
                
                # Parse and calculate
                result = self.evaluate_expression(expression)
                if result is not None:
                    # Store in history
                    self.history.append(f"{expression} = {result}")
                    print(f"Result: {result}")
                    
            except Exception as e:
                print(f"Calculation error: {e}")
    
    def evaluate_expression(self, expression):
        """Evaluate mathematical expressions with error handling"""
        try:
            # Split into components
            parts = re.split(r'\s+', expression)
            if len(parts) < 3:
                print("Invalid expression")
                return None
                
            num1 = float(parts[0])
            operator = parts[1]
            num2 = float(parts[2])
            
            # Perform calculation based on operator
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            elif operator == '*':
                result = num1 * num2
            elif operator == '/':
                if num2 == 0:
                    print("Error: Division by zero")
                    return None
                result = num1 / num2
            elif operator == '%':
                result = num1 % num2
            elif operator == '^':
                result = math.pow(num1, num2)
            else:
                print(f"Unsupported operator: {operator}")
                return None
                
            return result
        except Exception as e:
            print(f"Evaluation error: {e}")
            return None
    
    def scientific_calculator(self):
        """Scientific calculator with advanced functions"""
        print("\n--- SCIENTIFIC CALCULATOR ---")
        print("Available functions: sin, cos, tan, log, ln, sqrt, factorial")
        print("Format: function(number) or constant")
        print("Type 'back' to return to main menu")
        
        while True:
            try:
                user_input = input("\nEnter function: ").strip().lower()
                if user_input == 'back':
                    break
                
                result = self.evaluate_scientific_function(user_input)
                if result is not None:
                    self.history.append(f"{user_input} = {result}")
                    print(f"Result: {result}")
                    
            except Exception as e:
                print(f"Scientific calculation error: {e}")
    
    def evaluate_scientific_function(self, input_str):
        """Evaluate scientific functions"""
        try:
            # Handle constants
            if input_str == 'pi':
                return math.pi
            elif input_str == 'e':
                return math.e
            
            # Parse function calls
            match = re.match(r'^(\w+)\(([-+]?[0-9]*\.?[0-9]+)\)$', input_str)
            if not match:
                print("Invalid format. Use: function(number)")
                return None
                
            func_name = match.group(1)
            number = float(match.group(2))
            
            # Convert to radians for trigonometric functions
            if func_name in ['sin', 'cos', 'tan']:
                number = math.radians(number)
            
            # Perform calculation
            if func_name == 'sin':
                result = math.sin(number)
            elif func_name == 'cos':
                result = math.cos(number)
            elif func_name == 'tan':
                result = math.tan(number)
            elif func_name == 'log' and number > 0:
                result = math.log10(number)
            elif func_name == 'ln' and number > 0:
                result = math.log(number)
            elif func_name == 'sqrt' and number >= 0:
                result = math.sqrt(number)
            elif func_name == 'factorial' and number >= 0 and number == int(number):
                result = math.factorial(int(number))
            else:
                print(f"Unsupported function or invalid input: {func_name}")
                return None
                
            return result
        except Exception as e:
            print(f"Function evaluation error: {e}")
            return None
    
    def programming_calculator(self):
        """Calculator for programming-related conversions"""
        print("\n--- PROGRAMMING CALCULATOR ---")
        print("Convert between decimal, binary, octal, and hexadecimal")
        print("Type 'back' to return to main menu")
        
        while True:
            print("\n1. Decimal to other bases")
            print("2. Binary to other bases")
            print("3. Octal to other bases")
            print("4. Hexadecimal to other bases")
            print("5. Back to main menu")
            
            choice = input("Select conversion type (1-5): ").strip()
            
            if choice == '5' or choice.lower() == 'back':
                break
                
            if choice == '1':
                self.decimal_conversion()
            elif choice == '2':
                self.binary_conversion()
            elif choice == '3':
                self.octal_conversion()
            elif choice == '4':
                self.hexadecimal_conversion()
            else:
                print("Invalid choice")
    
    def decimal_conversion(self):
        """Convert decimal to other bases"""
        try:
            decimal = int(input("Enter decimal number: "))
            print(f"Binary: {bin(decimal)}")
            print(f"Octal: {oct(decimal)}")
            print(f"Hexadecimal: {hex(decimal)}")
            self.history.append(f"Decimal {decimal} converted")
        except ValueError:
            print("Invalid decimal number")
    
    def binary_conversion(self):
        """Convert binary to other bases"""
        try:
            binary = input("Enter binary number: ")
            decimal = int(binary, 2)
            print(f"Decimal: {decimal}")
            print(f"Octal: {oct(decimal)}")
            print(f"Hexadecimal: {hex(decimal)}")
            self.history.append(f"Binary {binary} converted")
        except ValueError:
            print("Invalid binary number")
    
    def octal_conversion(self):
        """Convert octal to other bases"""
        try:
            octal = input("Enter octal number: ")
            decimal = int(octal, 8)
            print(f"Decimal: {decimal}")
            print(f"Binary: {bin(decimal)}")
            print(f"Hexadecimal: {hex(decimal)}")
            self.history.append(f"Octal {octal} converted")
        except ValueError:
            print("Invalid octal number")
    
    def hexadecimal_conversion(self):
        """Convert hexadecimal to other bases"""
        try:
            hex_num = input("Enter hexadecimal number: ")
            decimal = int(hex_num, 16)
            print(f"Decimal: {decimal}")
            print(f"Binary: {bin(decimal)}")
            print(f"Octal: {oct(decimal)}")
            self.history.append(f"Hexadecimal {hex_num} converted")
        except ValueError:
            print("Invalid hexadecimal number")
    
    def financial_calculator(self):
        """Financial calculations for loans and investments"""
        print("\n--- FINANCIAL CALCULATOR ---")
        print("1. Simple Interest")
        print("2. Compound Interest")
        print("3. Loan Payment")
        print("4. Back to main menu")
        
        while True:
            choice = input("Select financial calculation (1-4): ").strip()
            
            if choice == '4' or choice.lower() == 'back':
                break
                
            if choice == '1':
                self.simple_interest()
            elif choice == '2':
                self.compound_interest()
            elif choice == '3':
                self.loan_payment()
            else:
                print("Invalid choice")
    
    def simple_interest(self):
        """Calculate simple interest"""
        try:
            principal = float(input("Enter principal amount: "))
            rate = float(input("Enter annual interest rate (%): "))
            time = float(input("Enter time in years: "))
            
            interest = principal * (rate / 100) * time
            total = principal + interest
            
            print(f"Simple Interest: {interest:.2f}")
            print(f"Total Amount: {total:.2f}")
            self.history.append(f"Simple Interest: P={principal}, R={rate}%, T={time} yrs")
        except ValueError:
            print("Invalid input")
    
    def compound_interest(self):
        """Calculate compound interest"""
        try:
            principal = float(input("Enter principal amount: "))
            rate = float(input("Enter annual interest rate (%): "))
            time = float(input("Enter time in years: "))
            compounds = int(input("Enter number of times compounded per year: "))
            
            amount = principal * math.pow(1 + (rate / 100) / compounds, compounds * time)
            interest = amount - principal
            
            print(f"Compound Interest: {interest:.2f}")
            print(f"Total Amount: {amount:.2f}")
            self.history.append(f"Compound Interest: P={principal}, R={rate}%, T={time} yrs")
        except ValueError:
            print("Invalid input")
    
    def loan_payment(self):
        """Calculate monthly loan payment"""
        try:
            principal = float(input("Enter loan amount: "))
            annual_rate = float(input("Enter annual interest rate (%): "))
            years = int(input("Enter loan term in years: "))
            
            monthly_rate = annual_rate / 100 / 12
            months = years * 12
            
            # Monthly payment formula
            if monthly_rate == 0:
                payment = principal / months
            else:
                payment = principal * (monthly_rate * math.pow(1 + monthly_rate, months)) / (math.pow(1 + monthly_rate, months) - 1)
            
            total_payment = payment * months
            total_interest = total_payment - principal
            
            print(f"Monthly Payment: {payment:.2f}")
            print(f"Total Payment: {total_payment:.2f}")
            print(f"Total Interest: {total_interest:.2f}")
            self.history.append(f"Loan Payment: Amount={principal}, Rate={annual_rate}%, Term={years} yrs")
        except ValueError:
            print("Invalid input")
    
    def statistics_calculator(self):
        """Calculate statistical measures"""
        print("\n--- STATISTICS CALCULATOR ---")
        print("Enter numbers separated by spaces")
        print("Type 'back' to return to main menu")
        
        while True:
            try:
                user_input = input("\nEnter numbers: ").strip()
                if user_input.lower() == 'back':
                    break
                
                # Convert input to list of numbers
                numbers = [float(x) for x in user_input.split()]
                
                if len(numbers) == 0:
                    print("No numbers entered")
                    continue
                
                # Calculate statistics
                count = len(numbers)
                total = sum(numbers)
                mean = total / count
                
                sorted_nums = sorted(numbers)
                median = self.calculate_median(sorted_nums)
                mode = self.calculate_mode(numbers)
                std_dev = self.calculate_std_dev(numbers, mean)
                
                # Display results
                print(f"Count: {count}")
                print(f"Sum: {total}")
                print(f"Mean: {mean:.4f}")
                print(f"Median: {median}")
                print(f"Mode: {mode}")
                print(f"Standard Deviation: {std_dev:.4f}")
                print(f"Min: {min(numbers)}")
                print(f"Max: {max(numbers)}")
                
                self.history.append(f"Statistics on {count} numbers")
                
            except ValueError:
                print("Invalid input. Please enter numbers only.")
            except Exception as e:
                print(f"Statistical calculation error: {e}")
    
    def calculate_median(self, sorted_nums):
        """Calculate median of sorted numbers"""
        n = len(sorted_nums)
        if n % 2 == 1:
            return sorted_nums[n // 2]
        else:
            return (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    
    def calculate_mode(self, numbers):
        """Calculate mode of numbers"""
        frequency = {}
        for num in numbers:
            frequency[num] = frequency.get(num, 0) + 1
        
        max_freq = max(frequency.values())
        modes = [num for num, freq in frequency.items() if freq == max_freq]
        
        return modes[0] if len(modes) == 1 else modes
    
    def calculate_std_dev(self, numbers, mean):
        """Calculate standard deviation"""
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        return math.sqrt(variance)
    
    def show_history(self):
        """Display calculation history"""
        print("\n--- CALCULATION HISTORY ---")
        if not self.history:
            print("No calculations in history")
            return
        
        for i, calculation in enumerate(self.history, 1):
            print(f"{i}. {calculation}")
        
        # Option to clear history
        clear = input("\nClear history? (y/n): ").lower()
        if clear == 'y':
            self.history.clear()
            print("History cleared")
    
    def memory_functions(self):
        """Memory operations"""
        print("\n--- MEMORY FUNCTIONS ---")
        print(f"Current Memory Value: {self.memory}")
        print("1. Store to Memory (M+)")
        print("2. Recall from Memory (MR)")
        print("3. Clear Memory (MC)")
        print("4. Add to Memory (M+)")
        print("5. Subtract from Memory (M-)")
        print("6. Back to main menu")
        
        while True:
            choice = input("Select memory operation (1-6): ").strip()
            
            if choice == '6' or choice.lower() == 'back':
                break
                
            if choice == '1':
                try:
                    value = float(input("Enter value to store: "))
                    self.memory = value
                    print(f"Stored {value} in memory")
                except ValueError:
                    print("Invalid value")
            elif choice == '2':
                print(f"Memory value: {self.memory}")
            elif choice == '3':
                self.memory = 0
                print("Memory cleared")
            elif choice == '4':
                try:
                    value = float(input("Enter value to add: "))
                    self.memory += value
                    print(f"Added {value}. Memory: {self.memory}")
                except ValueError:
                    print("Invalid value")
            elif choice == '5':
                try:
                    value = float(input("Enter value to subtract: "))
                    self.memory -= value
                    print(f"Subtracted {value}. Memory: {self.memory}")
                except ValueError:
                    print("Invalid value")
            else:
                print("Invalid choice")
    
    def run(self):
        """Main program loop"""
        print("Welcome to the Advanced Python Calculator!")
        print("This calculator demonstrates mastery of Python logic and loops.")
        
        while True:
            self.display_menu()
            choice = self.get_user_choice()
            
            if choice == 1:
                self.standard_calculator()
            elif choice == 2:
                self.scientific_calculator()
            elif choice == 3:
                self.programming_calculator()
            elif choice == 4:
                self.financial_calculator()
            elif choice == 5:
                self.statistics_calculator()
            elif choice == 6:
                self.show_history()
            elif choice == 7:
                self.memory_functions()
            elif choice == 8:
                print("Thank you for using the Advanced Python Calculator!")
                break

# Run the calculator
if __name__ == "__main__":
    calculator = AdvancedCalculator()
    calculator.run()