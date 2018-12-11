// import React from 'react';
import * as React from 'react';
// import ReactDOM from 'react-dom';
// import PropTypes from 'prop-types';
import Timeline from "./Timeline";

class Experiment extends React.Component {

  // props: Readonly<{}>;
  // state: Readonly<{}>;


  constructor(props: Readonly<{}>) {
    super(props);
  }

  // setState() {
  //
  // }

  render() {
    return (
      <div>
        <p>Hello, World!</p>
        <div className={"experiment"}>
          <Timeline/>
        </div>
      </div>
    );
  }
}

export default Experiment