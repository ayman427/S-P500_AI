// One gate covers selection changes and overlapping refreshes.
export function createRequestGate() {
  let selected = "";
  let generation = 0;
  return {
    select(selection: string) {
      selected = selection;
      generation += 1;
    },
    matches(selection: string) {
      return selection === selected;
    },
    begin(selection: string) {
      if (selection !== selected) return null;
      const request = ++generation;
      return () => request === generation && selection === selected;
    },
  };
}
