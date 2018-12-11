import * as React from 'react';
import WelcomeScreen from "./WelcomeScreen";

class SudokuExperiment extends React.Component {

  screen: number;

  constructor(props: Readonly<{}>) {
    super(props);
    this.screen = 0;
  }

  renderScreen() {
    if (this.screen == 0) {
      return <WelcomeScreen />
    }
    return <p>Error</p>
  }

  render() {
    return <div className={"SudokuExperiment"}>
      {this.renderScreen()}
    </div>;
  }
}

export default SudokuExperiment