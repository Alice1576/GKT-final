from classes import VectorizedDistillationColumn
from thermo import *

class GuessIterator:
    """
    :param feed: Feeden som kolonnen matas med
    :param column: Kolonnen som ska lösas

    Denna klass itererar fram lösningen till kolonnen genom att succesivt lösa svårare och svårare destillationer.
    Eftersom propan/propen är en enkel destillation att simulera förenklas feeden genom att allt vatten och väte tas bort.
    Kolonnen körs sedan för denna feed och lösningen till den simuleringen används som gissning för en lite svårare destillation
    där lite vatten och vätgas har lagts till. Till slut löses kolonnen för den riktiga feeden.
    """

    def __init__(self, feed: Stream, column: VectorizedDistillationColumn):
        self.feed = feed
        self.column = column

    def run(self, step=20):
        """
        :param step: antalet steg som iterationen ska utföras på. Sätts till 20 om inget annat anges.
        """
        feed_target_water: float = self.feed.flowrates["H2O"]
        feed_target_hydrogen: float = self.feed.flowrates["H2"]

        new_stream = Stream(
            temperature = self.feed.temperature,
            flowrates = self.feed.flowrates.copy(),
            pressure = self.feed.pressure,
            phase = self.feed.phase
        )

        previous_guess = None

        for j in np.linspace(0, 1, step):
            new_stream.flowrates["H2O"] = j*feed_target_water
            new_stream.flowrates["H2"] = j*feed_target_hydrogen

            _, _ = self.column.run(feed = new_stream, x0 = previous_guess)

            if not self.column.sol.success:
                print(f"Failed at j={j:.3f}: {self.column.sol.message}")
                break

            previous_guess = self.column.sol.x.flatten()

        return previous_guess