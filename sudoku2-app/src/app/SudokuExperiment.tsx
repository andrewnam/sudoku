import './SudokuExperiment.css';

import * as React from 'react';

import WelcomeScreen from "./WelcomeScreen";

import {Button} from 'primereact/button';

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
      <div className={"screen"}>
        {this.renderScreen()}
      </div>


      <div className={"next_button"}>
        <Button label="Next" className="p-button-success" />
      </div>
    </div>;
  }
}

export default SudokuExperiment