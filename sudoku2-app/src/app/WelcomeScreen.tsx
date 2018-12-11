import * as React from 'react';
import Screen from "../experiment/Screen";

class WelcomeScreen extends Screen {
  render() {
    return (
      <div className={"welcome_screen"}>
        Welcome to the experiment. Press the green right arrow button to continue.
      </div>
    );
  }
}

export default WelcomeScreen