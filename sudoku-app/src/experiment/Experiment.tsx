import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';
import Timeline from "./Timeline";

class Experiment extends React.Component {

  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div className={"experiment"}>
        <Timeline/>
      </div>
    );
  }
}

export default Experiment