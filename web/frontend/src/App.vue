<template>
<b-container fluid id="app">
  <SiteHeader/>
  <b-row id="main-pane">
    <b-col cols="4">
      <Sidebar/>
    </b-col>
    <b-col cols="8">
      <GraphView/>
    </b-col>

  </b-row>
</b-container>
</template>

<script>
import SiteHeader from './components/Header'
import Sidebar from './components/Sidebar'
import GraphView from './components/GraphView'

export default {
  name: 'app',
  components: {
    SiteHeader, Sidebar, GraphView
  }
}
</script>

<style>
#app {
  padding: 0;
}

h1, h2, h3, h4, h5, h6, p {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}

P {
  font-size: 12pt;
}
</style>
