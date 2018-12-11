import React from 'react';
// import ReactDOM from 'react-dom';

import Screen from "./Screen";

class Timeline extends React.Component {
  state: {
    timeline: Screen[],
    current_screen: number
  };

  constructor(props: Readonly<{}>) {
    super(props);
    this.state = {
      timeline: [],
      current_screen: 0
    };
  }

  render() {
    return (
      <div className={"timeline"}>
        Hello, World!
      </div>
    );
  }
}

export default Timeline