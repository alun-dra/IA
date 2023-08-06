<template>
  <div class="hello">
    <h1>probando</h1>
    <div>
      <form @submit.prevent="handleSubmit">
        <label for="simbolo">Simbolo:</label>
        <input type="text" id="simbolo" v-model="simbolo" required><br>
        
        <label for="forecast_out">Forecast Out:</label>
        <input type="number" id="forecast_out" v-model.number="forecastOut" required><br>
        
        <label for="fecha_inicio">Fecha de Inicio:</label>
        <input type="date" id="fecha_inicio" v-model="fechaInicio" required><br>
        
        <label for="fecha_termino">Fecha de Termino:</label>
        <input type="date" id="fecha_termino" v-model="fechaTermino" required><br>
        
        <button type="submit">Analizar</button>
      </form>
    
      <div v-if="submitted">
        <p>Simbolo: {{ simbolo }}</p>
        <p>Forecast Out: {{ forecastOut }}</p>
        <p>Fecha de Inicio: {{ fechaInicio }}</p>
        <p>Fecha de Termino: {{ fechaTermino }}</p>
      </div>
  </div>
    
  </div>
</template>

<script>
  export default {
    name: 'HomePage',
    data(){
      return{
        simbolo: '',
        forecastOut: 0,
        fechaInicio: '',
        fechaTermino: '',
        submitted: false

      };
    },
    methods:{
      handleSubmit(){
        this.submitted = true;
      }
    }

  }
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>

</style>
