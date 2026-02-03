import tkinter as tk
from tkinter import ttk, messagebox, font
import random

# --- Card Data Tailored for Long-Term Couples ---
CARDS_LONG_TERM = {
    "Reflection & Growth": [
        "Looking back, what's a moment you realized our love had deepened or evolved in a significant way?",
        "What's one way our individual growth has positively impacted our relationship?",
        "If our relationship was a book, what would the current chapter be titled, and why?",
        "What's a quality you admire in me now that you perhaps didn't fully appreciate in our early years?",
        "How have our shared challenges made us stronger as a couple?",
        "What's a dream we once had that we've achieved, and how does that make you feel?",
        "In what ways do you feel we still surprise each other?",
        "What's one lesson about love that our relationship has taught you?",
        "How can we better support each other's personal passions or hobbies in this phase of our lives?",
        "What's a 'small thing' that consistently makes you feel loved and seen by me?"
    ],
    "Connection & Intimacy": [
        "What's a non-verbal cue you use that you wish I understood better, or one I use that you cherish?",
        "Describe a perfect 'us' evening, with no distractions.",
        "What's one way we can intentionally cultivate more joy or playfulness in our daily lives together?",
        "If we could relive one specific day from our past together, which would it be and why?",
        "What's something new you'd like us to learn or experience together in the coming year?",
        "How can we create more moments of quiet, undistracted connection?",
        "What does 'intimacy' mean to you at this stage of our relationship, beyond the physical?",
        "Share a favorite memory of us just laughing uncontrollably.",
        "What's a compliment you've always wanted to give me but perhaps haven't fully expressed?",
        "How can I make you feel more desired and appreciated on a regular basis?"
    ],
    "Future & Dreams": [
        "If we could write a 'mission statement' for our relationship for the next decade, what would it say?",
        "What's a new tradition you'd like us to start together, or an old one to revisit?",
        "As we look to the future, what are you most excited about for us as a couple?",
        "What's a shared adventure (big or small) you're dreaming of for our 'empty nest' or 'retirement' years (even if far off)?",
        "How do you envision us supporting each other through future life changes or challenges?",
        "What kind of legacy do you hope our relationship leaves for others (e.g., children, friends)?",
        "If we could master one new skill together in the next five years, what would it be and why?",
        "What does 'growing old together' look like in your ideal vision?",
        "What's one thing you hope we *never* stop doing together?",
        "How can we ensure our relationship continues to be a source of inspiration and comfort for each other?"
    ],
    "Playful Challenge": [
        "Without speaking, spend 3 minutes showing your partner appreciation through touch (e.g., holding hands, a back rub).",
        "Choose a song that reminds you of your partner or your relationship. Play it and share why you chose it.",
        "Write down 3 things you're grateful for about your partner. Take turns reading one at a time.",
        "Give your partner a genuine, heartfelt compliment that goes beyond their physical appearance.",
        "Recreate a favorite photo of you two, or take a new one capturing your current connection.",
        "Spend 60 seconds looking into each other's eyes, then share one word that describes how it made you feel.",
        "Plan a simple, spontaneous 'micro-date' to happen in the next 24 hours (e.g., a walk, coffee, stargazing).",
        "Each of you secretly write down your favorite shared memory. Reveal and discuss.",
        "Offer your partner a 2-minute temple or hand massage.",
        "Tell your partner a story about them that always makes you smile, or one that showcases a quality you love."
    ]
}

# --- GUI Application ---

class ConnectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CONNECT - The Couple's Game (Long-Term Edition)")
        self.root.geometry("800x650") # Adjusted size
        self.root.configure(bg="#2C3E50") # Dark blue-grey background

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'

        # Custom Colors
        self.primary_color = "#3498DB"  # Bright Blue
        self.secondary_color = "#2ECC71" # Green
        self.text_color = "#ECF0F1"     # Light Grey/White
        self.card_bg_color = "#34495E"  # Darker blue-grey for card
        self.button_hover_color = "#2980B9"

        self.style.configure("TFrame", background=self.root["bg"])
        self.style.configure("TLabel", background=self.root["bg"], foreground=self.text_color, font=("Helvetica", 12))
        self.style.configure("Title.TLabel", font=("Helvetica", 24, "bold"))
        self.style.configure("Category.TButton", font=("Helvetica", 14, "bold"), padding=10,
                             background=self.primary_color, foreground="white", relief="flat",
                             borderwidth=0)
        self.style.map("Category.TButton",
                       background=[('active', self.button_hover_color)],
                       relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        self.style.configure("Action.TButton", font=("Helvetica", 12), padding=8,
                             background=self.secondary_color, foreground="white")
        self.style.map("Action.TButton",
                       background=[('active', "#27AE60")]) # Darker green on hover

        self.style.configure("Card.TFrame", background=self.card_bg_color, relief="raised", borderwidth=5, padding=15)
        self.style.configure("CardText.TLabel", background=self.card_bg_color, foreground=self.text_color,
                             font=("Georgia", 16, "italic"), wraplength=500) # Georgia for a bit more elegance

        self.player1_name = "Partner 1"
        self.player2_name = "Partner 2"
        self.current_player_name = ""
        self.other_player_name = ""
        self.player_turn_is_p1 = True # True if P1's turn to pick, False if P2's
        self.answer_stage = 0 # 0: pick category, 1: P_draw answers, 2: P_other answers

        self.decks = {}
        self.current_card_text = ""
        self.current_category = ""

        self.setup_name_input_screen()

    def setup_name_input_screen(self):
        self.clear_window()
        self.name_frame = ttk.Frame(self.root, padding=20)
        self.name_frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(self.name_frame, text="Welcome to CONNECT!", style="Title.TLabel").pack(pady=20)
        ttk.Label(self.name_frame, text="Enter your names to begin:", font=("Helvetica", 14)).pack(pady=10)

        ttk.Label(self.name_frame, text="Partner 1 Name:").pack(pady=5)
        self.p1_name_entry = ttk.Entry(self.name_frame, font=("Helvetica", 12), width=30)
        self.p1_name_entry.pack(pady=5)
        self.p1_name_entry.insert(0, "Partner 1")

        ttk.Label(self.name_frame, text="Partner 2 Name:").pack(pady=5)
        self.p2_name_entry = ttk.Entry(self.name_frame, font=("Helvetica", 12), width=30)
        self.p2_name_entry.pack(pady=5)
        self.p2_name_entry.insert(0, "Partner 2")

        start_button = ttk.Button(self.name_frame, text="Start Game", command=self.initialize_game_with_names, style="Action.TButton")
        start_button.pack(pady=20)

    def initialize_game_with_names(self):
        p1 = self.p1_name_entry.get().strip()
        p2 = self.p2_name_entry.get().strip()
        if p1: self.player1_name = p1
        if p2: self.player2_name = p2

        self.create_decks()
        self.setup_main_game_ui()
        self.update_turn_indicator()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_decks(self):
        self.decks = {}
        for category, questions in CARDS_LONG_TERM.items():
            deck = list(questions)
            random.shuffle(deck)
            self.decks[category] = deck

    def setup_main_game_ui(self):
        self.clear_window()

        # Main Frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        main_frame.grid_columnconfigure(0, weight=1) # Make column 0 expandable

        # Turn Indicator
        self.turn_label = ttk.Label(main_frame, text="", font=("Helvetica", 16, "bold"))
        self.turn_label.grid(row=0, column=0, pady=10, sticky="ew")

        # Category Selection Frame
        self.category_frame = ttk.Frame(main_frame)
        self.category_frame.grid(row=1, column=0, pady=15, sticky="ew")
        self.category_frame.grid_columnconfigure((0,1), weight=1) # Make columns responsive

        row, col = 0, 0
        self.category_buttons = {}
        for i, category_name in enumerate(self.decks.keys()):
            btn = ttk.Button(self.category_frame, text=f"{category_name} ({len(self.decks[category_name])})",
                             command=lambda c=category_name: self.category_chosen(c), style="Category.TButton")
            btn.grid(row=row, column=col, padx=10, pady=10, sticky="ew")
            self.category_buttons[category_name] = btn
            col += 1
            if col > 1: # Max 2 buttons per row
                col = 0
                row += 1

        # Card Display Frame
        self.card_display_frame = ttk.Frame(main_frame, style="Card.TFrame")
        self.card_display_frame.grid(row=2, column=0, pady=20, sticky="nsew")
        main_frame.grid_rowconfigure(2, weight=1) # Make card display area expand

        self.card_label = ttk.Label(self.card_display_frame, text="Choose a category to draw a card.",
                                   style="CardText.TLabel", justify=tk.CENTER)
        self.card_label.pack(expand=True, padx=20, pady=20)


        # Instruction Label
        self.instruction_label = ttk.Label(main_frame, text="", font=("Helvetica", 12, "italic"), wraplength=700, justify=tk.CENTER)
        self.instruction_label.grid(row=3, column=0, pady=10, sticky="ew")

        # Action Button
        self.action_button = ttk.Button(main_frame, text="Next Step", command=self.handle_action, style="Action.TButton")
        self.action_button.grid(row=4, column=0, pady=20, sticky="ew")
        self.action_button.grid_remove() # Hide initially

    def update_turn_indicator(self):
        if self.player_turn_is_p1:
            self.current_player_name = self.player1_name
            self.other_player_name = self.player2_name
        else:
            self.current_player_name = self.player2_name
            self.other_player_name = self.player1_name

        if self.answer_stage == 0: # Category selection phase
            self.turn_label.config(text=f"{self.current_player_name}'s Turn to Choose a Category")
            self.instruction_label.config(text="")
            self.card_label.config(text="Select a category above.")
            self.enable_category_buttons()
            self.action_button.grid_remove()
        elif self.answer_stage == 1: # Drawer answers
            self.turn_label.config(text=f"Card for: {self.current_player_name}")
            self.instruction_label.config(text=f"{self.current_player_name}, please read and answer/do the challenge first.")
            self.action_button.config(text=f"{self.current_player_name} Finished")
            self.action_button.grid()
        elif self.answer_stage == 2: # Other player answers (if not a solo challenge)
            self.turn_label.config(text=f"Now for: {self.other_player_name}")
            self.instruction_label.config(text=f"{self.other_player_name}, now it's your turn to respond/participate.")
            self.action_button.config(text=f"{self.other_player_name} Finished")
            self.action_button.grid()


    def enable_category_buttons(self, enable=True):
        for category, button in self.category_buttons.items():
            if self.decks[category]: # Only enable if cards are left
                button.config(state=tk.NORMAL if enable else tk.DISABLED)
            else:
                button.config(text=f"{category} (Empty)", state=tk.DISABLED)


    def category_chosen(self, category_name):
        if not self.decks[category_name]:
            messagebox.showinfo("Empty Deck", f"The '{category_name}' deck is empty. Please choose another.")
            return

        self.current_category = category_name
        self.current_card_text = self.decks[category_name].pop()
        self.category_buttons[category_name].config(text=f"{category_name} ({len(self.decks[category_name])})")

        self.card_label.config(text=f"\"{self.current_card_text}\"")
        self.answer_stage = 1
        self.enable_category_buttons(False) # Disable category buttons during answering
        self.update_turn_indicator()

        if not self.decks[category_name]: # If deck became empty
            self.category_buttons[category_name].config(state=tk.DISABLED)

        if self.check_game_over():
            return


    def handle_action(self):
        if self.answer_stage == 1: # Current player (drawer) finished
            # For challenges, sometimes only one person acts, or it's simultaneous.
            # For questions, the other person always gets a turn.
            is_challenge = self.current_category == "Playful Challenge"
            is_solo_challenge = is_challenge and ("your partner" not in self.current_card_text.lower() and
                                                  "each other" not in self.current_card_text.lower() and
                                                  "together" not in self.current_card_text.lower())

            if not is_challenge or not is_solo_challenge:
                self.answer_stage = 2 # Move to other player's turn to answer/participate
            else: # Solo challenge or simultaneous challenge done
                self.player_turn_is_p1 = not self.player_turn_is_p1 # Next player's turn to pick
                self.answer_stage = 0
            self.update_turn_indicator()

        elif self.answer_stage == 2: # Other player finished
            self.player_turn_is_p1 = not self.player_turn_is_p1 # Next player's turn to pick
            self.answer_stage = 0
            self.update_turn_indicator()

        if self.check_game_over():
            return


    def check_game_over(self):
        if all(not deck for deck in self.decks.values()):
            self.card_label.config(text="All cards drawn! Game Over.\nWe hope you connected deeply!")
            self.instruction_label.config(text="Thank you for playing!")
            self.turn_label.config(text="Game Finished!")
            self.action_button.grid_remove()
            self.enable_category_buttons(False)
            play_again_button = ttk.Button(self.root, text="Play Again?", command=self.setup_name_input_screen, style="Action.TButton")
            play_again_button.pack(pady=20) # Or use grid
            return True
        return False


if __name__ == "__main__":
    root = tk.Tk()
    app = ConnectApp(root)
    root.mainloop()