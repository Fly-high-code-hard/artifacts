# Идея, проблема и потенциальное решение


<table>
  <tr>
    <th>Проблематика</th>
    <th>Потенциальные пользователи</th>
    <th>Решение</th>
    <th>Конкуренты</th>
    <th>Уникальность решения</th>
    <th>Ссылка на видео с демонстрацией продукта</th>
  </tr>
  <tr>
    <td style="vertical-align: top; ">Проблема - задержка рейсов, которая приводит к выплате компенсаций, недовольству пассажиров.  Система разрабатывается для работников S7, чтобы улучшить лояльность клиентов и снизить издержки компании. Что значит снизить издержки? Авиакомпании должны предоставлять пассажирам питание, жилье, гостиницы, трансфер до гостиниц, денежные компенсации в случае задержек рейса. Это деньги. Если мы сможем повысить точность и надежность прогнозов, то объем компенсаций станет ниже, так как потенциально снижает время, на которое будет отложен рейс. (Рейс по прогнозам компании задержится на 7 часов, рейс отложен на 7 часов. На самом деле, при  более точном прогнозе рейс мог бы быть задержан на 5 часов)  Либо же, наоборот, прогноз компании оказался ниже реального. Это еще хуже, так как компенсации по закону будут выше и недовольство пассажиров кратно увеличится. Однако, это не единственная причина, по которой мы разработали нашу систему. Она будет очень полезна самой авиакомпании при планировании расписания, т.е. составления его таким образом, что задержки в полетах будут минимальны - насколько это возможно, конечно.</td>
    <td style="vertical-align: top; ">Заинтересованные стороны - cотрудники S7 и ее клиенты (авиапассажиры, летающие компанией S7). Компания заинтересована в снижении издержек и повышении лояльности клиентов. Клиенты хотят точно знать, насколько задержится их рейс заранее. Также может потребоваться оператор (в будущем - может быть заменен на автоматическую систему ввода) вводит данные о рейсе и получает информацию о том, какая будет задержка. Система потом отправляет всем пассажирам рейса информацию о задержке и компенсациях, полагаемых им, помогает авиакомпании оптимальнее планировать расписание.</td>
     <td style="vertical-align: top;">Идея - создать систему, предсказывающую задержки рейсов. Предсказания будут основываться, исходя из различных признаков: погодные условия, время вылета, модель самолета и др. стандартные признаки. Результат работы алгоритма предсказания - класс задержки (см. в DESC классы задержек), к которому вероятнее всего будет относиться рейс при известной информации о нем. Также система будет выводить информацию о типе компенсации, которая будет полагаться пассажирам при таком классе задержки.</td>
    <td style="vertical-align: top;">Существующие альтернативы в большинстве своем также направлены на прогнозирование возможных ошибок и проблем:<br/> 
      1. Прогнозирование и предотвращение технических неисправностей – Внедрение предиктивного технического обслуживания с помощью IoT и сенсоров, которые позволяют выявлять потенциальные неисправности до их возникновения. Это снижает вероятность технических задержек и повышает общую пунктуальность рейсов.<br/> Также существуют альтернативы, направленные на динамическую оптимизацию условий, минимизирущих проблемы и ошибки:<br/> 2. Динамическое управление экипажами и ресурсами – Использование автоматизированных систем для оперативного перераспределения пилотов, бортпроводников и наземного персонала в случае задержек. Это позволяет минимизировать задержки, связанные с нехваткой экипажа или сменой воздушного судна.</td>
    <td style="vertical-align: top;">Мы предлагаем не просто прогнозирование задержек, а интеллектуальную систему, которая заранее определяет вероятность сбоев и позволяет авиакомпаниям минимизировать их влияние. Наша платформа помогает не только предупреждать задержки, но и заранее рассчитывать возможные компенсации, что снижает финансовые риски и повышает лояльность пассажиров. В результате авиакомпании получают мощный инструмент для оптимизации работы и улучшения клиентского опыта. Кроме того, наша система постоянно самообучается, адаптируясь к новым условиям. Такой уровень интеграции данных, точность прогнозирования и гибкость модели делают наше решение сложным для копирования конкурентами.</td>
     <td style="vertical-align: top;">(https://drive.google.com/file/d/13kOmdRRCVa7QQcR_3Q8iWu1ERmA631H1/view?usp=sharing)</td>
  </tr>
</table>
